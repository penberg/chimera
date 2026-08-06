//! The prompt Chimera gives the shell it starts on its own.
//!
//! An interactive session must not look like the shell that launched it: the
//! guest's writes land in a filesystem's delta, or, under `--unsafe`, on the
//! host itself. Bash takes its prompt setup from an rc file, so Chimera writes
//! one naming the filesystem that receives the session's writes and hands it
//! over as `--rcfile`.
//!
//! Injecting an rc file, rather than exporting `PS1` or `PROMPT_COMMAND`, is
//! what makes the badge survive: the file replaces the rc entry point and
//! sources the user's own `.bashrc` from inside, so a prompt set there cannot
//! overwrite the badge the way it overwrites an inherited variable.

use std::{
    env,
    fs::{self, DirBuilder},
    io,
    os::unix::fs::DirBuilderExt,
    path::PathBuf,
    process,
};

use crate::fs::{Filesystem, fresh_id};

/// How much of a filesystem name the badge shows. A generated id is far
/// shorter; an attach selector naming a path can be arbitrarily long, and the
/// prompt is a label, not a location.
const BADGE_MAX: usize = 16;

/// A session's rc file, and the directory that holds it. Dropping this removes
/// both, so the value lives as long as the guest may read the file.
pub struct Prompt {
    dir: PathBuf,
    /// The process that created the directory. Guest fork is a host fork, so
    /// every child of the guest tree returns through the CLI carrying a copy
    /// of this struct; only the creator may remove a tree the rest of the
    /// session is still using.
    owner: u32,
}

impl Prompt {
    /// Write the rc file for a shell Chimera started itself. It sources the
    /// user's `.bashrc` and then badges the prompt with `fsys`, or with
    /// `unsafe` when there is no filesystem and writes reach the host.
    pub fn new(fsys: Option<&Filesystem>) -> io::Result<Prompt> {
        let base = env::temp_dir();
        loop {
            let dir = base.join(format!("chimera-prompt-{}", fresh_id()?));
            // An unguessable name created exclusively, in a directory no one
            // else may write: Bash runs what this holds as the user, so a
            // predictable path on a shared `/tmp` would be an invitation to
            // plant the rc file first.
            match DirBuilder::new().mode(0o700).create(&dir) {
                Ok(()) => {
                    let prompt = Prompt {
                        dir,
                        owner: process::id(),
                    };
                    fs::write(prompt.rcfile(), script(fsys))?;
                    return Ok(prompt);
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// The path to hand Bash as `--rcfile`. The guest reads it through its own
    /// namespace, where the host's `/tmp` shows through the overlay's lower
    /// layer unchanged.
    pub fn rcfile(&self) -> PathBuf {
        self.dir.join("bashrc")
    }
}

impl Drop for Prompt {
    fn drop(&mut self) {
        if process::id() == self.owner {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

fn script(fsys: Option<&Filesystem>) -> String {
    let (color, badge) = match fsys {
        // Reverse video reads as a badge under both light and dark themes.
        Some(fsys) => ("7", badge(fsys)),
        // Writes reach the host, which is worth the one alarming color.
        None => ("97;41", "unsafe".to_string()),
    };
    format!(
        r#"if [[ -n ${{HOME-}} && -r $HOME/.bashrc ]]; then
    builtin source "$HOME/.bashrc"
fi

if [[ ${{TERM-}} != dumb && -z ${{NO_COLOR+x}} ]]; then
    PS1='\[\e[{color}m\] {badge} \[\e[0m\] \w \$ '
else
    PS1='[{badge}] \w \$ '
fi
"#
    )
}

/// The filesystem's name as it can stand inside the single-quoted `PS1`
/// assignment: an attach selector carries whatever the user typed, so keep
/// the portable filename characters and replace everything else.
fn badge(fsys: &Filesystem) -> String {
    let name = fsys.root.file_name().unwrap_or(fsys.id.as_ref());
    let badge: String = name
        .to_string_lossy()
        .chars()
        .take(BADGE_MAX)
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '?'
            }
        })
        .collect();
    if badge.is_empty() { "?".into() } else { badge }
}
