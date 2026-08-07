//! The prompt Chimera gives the shell it starts on its own.
//!
//! An interactive session must not look like the shell that launched it: the
//! guest's writes land in a filesystem's delta, or — under `--unsafe`, and on
//! any host without the copy-on-write filesystem — on the host itself. The
//! badge rides the shell's own startup mechanism: Bash takes an rc file over
//! `--rcfile`, and zsh reads its startup files from `$ZDOTDIR`.
//!
//! Entering through the startup files, rather than exporting `PS1` or
//! `PROMPT_COMMAND`, is what makes the badge survive: each stub sources the
//! user's own rc file from inside and sets the prompt after it returns, so a
//! prompt set there cannot overwrite the badge the way it overwrites an
//! inherited variable.

use std::{
    env,
    fs::{self, DirBuilder},
    io,
    os::unix::fs::DirBuilderExt,
    path::PathBuf,
    process,
};

#[cfg(not(target_os = "linux"))]
use std::path::Path;

#[cfg(target_os = "linux")]
use crate::fs::Filesystem;
use crate::fs::fresh_id;

/// How much of a filesystem name the badge shows. A generated id is far
/// shorter; an attach selector naming a path can be arbitrarily long, and the
/// prompt is a label, not a location.
#[cfg(target_os = "linux")]
const BADGE_MAX: usize = 16;

/// The shells whose startup the badge knows how to enter. Anything else runs
/// unbadged rather than mis-badged.
pub enum Shell {
    Bash,
    // The overlay path badges the bash it starts itself; a host without the
    // filesystem badges whatever `$SHELL` names, zsh being its default.
    #[cfg(not(target_os = "linux"))]
    Zsh,
}

/// Which badge mechanism `program` understands, by its basename.
#[cfg(not(target_os = "linux"))]
pub fn shell_kind(program: &Path) -> Option<Shell> {
    match program.file_name()?.to_str()? {
        "bash" => Some(Shell::Bash),
        "zsh" => Some(Shell::Zsh),
        _ => None,
    }
}

/// A session's startup stubs, and the directory that holds them. Dropping
/// this removes both, so the value lives as long as the guest may read them.
pub struct Prompt {
    dir: PathBuf,
    /// The process that created the directory. Guest fork is a host fork, so
    /// every child of the guest tree returns through the CLI carrying a copy
    /// of this struct; only the creator may remove a tree the rest of the
    /// session is still using.
    owner: u32,
}

impl Prompt {
    /// Write the startup stubs for a shell Chimera started itself. Each
    /// sources the user's own startup file and then badges the prompt with
    /// `badge`, or with `unsafe` when there is no filesystem and writes reach
    /// the host.
    pub fn new(shell: &Shell, badge: Option<&str>) -> io::Result<Prompt> {
        let base = env::temp_dir();
        loop {
            let dir = base.join(format!("chimera-prompt-{}", fresh_id()?));
            // An unguessable name created exclusively, in a directory no one
            // else may write: the shell runs what this holds as the user, so a
            // predictable path on a shared `/tmp` would be an invitation to
            // plant the startup file first.
            match DirBuilder::new().mode(0o700).create(&dir) {
                Ok(()) => {
                    let prompt = Prompt {
                        dir,
                        owner: process::id(),
                    };
                    match shell {
                        Shell::Bash => fs::write(prompt.rcfile(), bash_script(badge))?,
                        #[cfg(not(target_os = "linux"))]
                        Shell::Zsh => {
                            // zsh reads `$ZDOTDIR/.zshenv` for every shell and
                            // `$ZDOTDIR/.zshrc` for an interactive one; both
                            // stubs exist so redirecting `ZDOTDIR` costs the
                            // user none of their own startup.
                            fs::write(prompt.dir.join(".zshenv"), ZSHENV)?;
                            fs::write(prompt.dir.join(".zshrc"), zsh_script(badge))?;
                        }
                    }
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

    /// The directory to hand zsh as `$ZDOTDIR`.
    #[cfg(not(target_os = "linux"))]
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for Prompt {
    fn drop(&mut self) {
        if process::id() == self.owner {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

/// The badge and its color: reverse video reads as a badge under both light
/// and dark themes, and `None` — writes reach the host — is worth the one
/// alarming color.
fn style(badge: Option<&str>) -> (&'static str, String) {
    match badge {
        Some(badge) => ("7", badge.to_string()),
        None => ("97;41", "unsafe".to_string()),
    }
}

fn bash_script(badge: Option<&str>) -> String {
    let (color, badge) = style(badge);
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

#[cfg(not(target_os = "linux"))]
const ZSHENV: &str = r#"if [[ -n ${HOME-} && -r $HOME/.zshenv ]]; then
    builtin source "$HOME/.zshenv"
fi
"#;

#[cfg(not(target_os = "linux"))]
fn zsh_script(badge: Option<&str>) -> String {
    let (color, badge) = style(badge);
    // zsh does not expand `\e` inside a single-quoted PS1, so the escape
    // byte is written literally, wrapped in `%{ %}` to mark it zero-width.
    format!(
        r#"if [[ -n ${{HOME-}} && -r $HOME/.zshrc ]]; then
    builtin source "$HOME/.zshrc"
fi

if [[ ${{TERM-}} != dumb && -z ${{NO_COLOR+x}} ]]; then
    PS1='%{{{esc}[{color}m%}} {badge} %{{{esc}[0m%}} %~ %# '
else
    PS1='[{badge}] %~ %# '
fi
"#,
        esc = '\x1b',
    )
}

/// The filesystem's name as it can stand inside the single-quoted `PS1`
/// assignment: an attach selector carries whatever the user typed, so keep
/// the portable filename characters and replace everything else.
#[cfg(target_os = "linux")]
pub fn filesystem_badge(fsys: &Filesystem) -> String {
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
