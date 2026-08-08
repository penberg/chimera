//! `docker:` image pull: fetch a container image from an OCI registry and
//! keep it as an image filesystem — a complete base tree with no parent.
//!
//! The pull flattens the image's layer stack at fetch time: each layer tar is
//! applied in order onto one staged tree, OCI whiteouts included, so by the
//! time the filesystem exists the runtime never sees layers at all. The tree
//! then serves as an overlay *lower*, exactly the role the live host plays
//! for an ordinary filesystem, which is why nothing downstream of the pull
//! needs to know images exist.
//!
//! Transport is `curl(1)` and digests are checked with `sha256sum(1)`: the
//! registry protocol is a handful of HTTPS GETs, and borrowing the system
//! tools keeps a TLS stack out of Chimera's build. Only the anonymous Bearer
//! token flow is spoken, which covers public images on Docker Hub, GHCR, and
//! any other distribution-spec registry.

use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::Command,
};

use crate::fs::{adopt_image_root, find_image_root, staging_dir};

/// A parsed `docker:` locator, normalized the way Docker normalizes
/// references: a bare name gains `docker.io/library/`, a missing tag reads as
/// `latest`. `reference` is what the manifest is fetched by — a tag, or a
/// `sha256:` digest when the locator pinned one.
struct ImageRef {
    /// Registry host as named in the reference (`docker.io`, `ghcr.io`, …).
    registry: String,
    repository: String,
    reference: String,
    /// The normalized locator, e.g. `docker:docker.io/library/debian:13-slim`
    /// — the identity two differently-typed locators for the same image agree
    /// on, recorded in provenance and matched on re-pull.
    canonical: String,
}

impl ImageRef {
    fn parse(locator: &str) -> io::Result<ImageRef> {
        let Some(rest) = locator.strip_prefix("docker:") else {
            return Err(invalid(format!(
                "not a docker: locator: {locator} (try docker:{locator})"
            )));
        };
        let (name, digest) = match rest.split_once('@') {
            Some((name, digest)) => {
                if !digest.starts_with("sha256:") {
                    return Err(invalid(format!("unsupported digest in {locator}")));
                }
                (name, Some(digest))
            }
            None => (rest, None),
        };
        // Docker's own registry heuristic: the first component names a
        // registry only when it can't be a repository name — it contains a
        // dot or a port, or is exactly `localhost`.
        let (registry, path) = match name.split_once('/') {
            Some((first, path))
                if first.contains('.') || first.contains(':') || first == "localhost" =>
            {
                (first.to_string(), path.to_string())
            }
            _ => ("docker.io".to_string(), name.to_string()),
        };
        // The tag is the part after a colon in the *last* component; earlier
        // colons can only be the registry's port, consumed above.
        let (repository, tag) = match path.rsplit_once(':') {
            Some((repo, tag)) if !tag.contains('/') => (repo.to_string(), tag.to_string()),
            _ => (path, "latest".to_string()),
        };
        if repository.is_empty() {
            return Err(invalid(format!("no repository in {locator}")));
        }
        let repository = if registry == "docker.io" && !repository.contains('/') {
            format!("library/{repository}")
        } else {
            repository
        };
        let canonical = match digest {
            Some(digest) => format!("docker:{registry}/{repository}@{digest}"),
            None => format!("docker:{registry}/{repository}:{tag}"),
        };
        Ok(ImageRef {
            registry,
            repository,
            reference: digest.map(str::to_string).unwrap_or(tag),
            canonical,
        })
    }

    /// The API host: Docker Hub's registry lives at `registry-1.docker.io`,
    /// its user-facing name notwithstanding.
    fn api_host(&self) -> &str {
        if self.registry == "docker.io" {
            "registry-1.docker.io"
        } else {
            &self.registry
        }
    }

    fn pinned_digest(&self) -> Option<&str> {
        self.reference
            .starts_with("sha256:")
            .then_some(self.reference.as_str())
    }
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Resolve a `docker:` locator to an image filesystem's id, pulling the image
/// if no kept filesystem matches it. The match is by digest when the locator
/// pins one and by normalized reference otherwise, so re-running a `--from`
/// costs no network; refreshing a moved tag is a `fs rm` of the old root and
/// a new pull.
pub fn pull(locator: &str) -> io::Result<String> {
    let image = ImageRef::parse(locator)?;
    if let Some(id) = find_image_root(&image.canonical, image.pinned_digest()) {
        return Ok(id);
    }
    eprintln!("chimera: pulling {}", image.canonical);
    let stage = Staging::new()?;
    let (digest, layers) = fetch_manifest(&image)?;
    let data = stage.dir.join("data");
    fs::create_dir(&data)?;
    for (i, layer) in layers.iter().enumerate() {
        eprintln!(
            "chimera:   layer {}/{} ({})",
            i + 1,
            layers.len(),
            human_size(layer.size),
        );
        let blob = stage.dir.join("layer");
        fetch_blob(&image, &layer.digest, &blob)?;
        apply_layer(&blob, &layer.media_type, &data)?;
        fs::remove_file(&blob)?;
    }
    let id = adopt_image_root(&stage.dir, locator, &image.canonical, &digest)?;
    eprintln!("chimera: pulled {} as {id}", image.canonical);
    Ok(id)
}

/// The staging directory a pull assembles the tree in, removed on failure —
/// an aborted pull must not leave a half-image where the next run finds it.
struct Staging {
    dir: PathBuf,
    keep: bool,
}

impl Staging {
    fn new() -> io::Result<Staging> {
        let base = staging_dir();
        fs::create_dir_all(&base)?;
        loop {
            let dir = base.join(crate::fs::fresh_id()?);
            match fs::create_dir(&dir) {
                Ok(()) => return Ok(Staging { dir, keep: false }),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

struct Layer {
    digest: String,
    media_type: String,
    size: u64,
}

/// Fetch the image's manifest, stepping through a multi-platform index to the
/// linux/amd64 entry, and return the manifest digest with the layer list.
fn fetch_manifest(image: &ImageRef) -> io::Result<(String, Vec<Layer>)> {
    const ACCEPT: &str = "application/vnd.oci.image.index.v1+json, \
         application/vnd.docker.distribution.manifest.list.v2+json, \
         application/vnd.oci.image.manifest.v1+json, \
         application/vnd.docker.distribution.manifest.v2+json";

    let token = token(image)?;
    let get = |reference: &str| -> io::Result<(serde_json::Value, String)> {
        let url = format!(
            "https://{}/v2/{}/manifests/{}",
            image.api_host(),
            image.repository,
            reference,
        );
        let body = http_get(&url, ACCEPT, token.as_deref(), None)?;
        let digest = format!("sha256:{}", sha256(&body)?);
        if reference.starts_with("sha256:") && digest != reference {
            return Err(io::Error::other(format!(
                "manifest digest mismatch for {url}: got {digest}"
            )));
        }
        let json = serde_json::from_slice(&body)
            .map_err(|e| io::Error::other(format!("bad manifest from {url}: {e}")))?;
        Ok((json, digest))
    };

    let (mut manifest, mut digest) = get(&image.reference)?;
    if manifest.get("layers").is_none() {
        // A multi-platform index: descend into the linux/amd64 manifest. Its
        // digest is the identity recorded — the platform image is what the
        // filesystem holds.
        let entry = manifest["manifests"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|m| {
                m["platform"]["os"].as_str() == Some("linux")
                    && m["platform"]["architecture"].as_str() == Some("amd64")
            })
            .and_then(|m| m["digest"].as_str())
            .ok_or_else(|| io::Error::other(format!("{}: no linux/amd64 image", image.canonical)))?
            .to_string();
        (manifest, digest) = get(&entry)?;
    }
    let layers = manifest["layers"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|l| {
            Ok(Layer {
                digest: l["digest"]
                    .as_str()
                    .ok_or_else(|| io::Error::other("layer without digest"))?
                    .to_string(),
                media_type: l["mediaType"].as_str().unwrap_or_default().to_string(),
                size: l["size"].as_u64().unwrap_or(0),
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    if layers.is_empty() {
        return Err(io::Error::other(format!(
            "{}: manifest names no layers",
            image.canonical
        )));
    }
    Ok((digest, layers))
}

fn fetch_blob(image: &ImageRef, digest: &str, out: &Path) -> io::Result<()> {
    let token = token(image)?;
    let url = format!(
        "https://{}/v2/{}/blobs/{digest}",
        image.api_host(),
        image.repository,
    );
    http_get(
        &url,
        "application/octet-stream",
        token.as_deref(),
        Some(out),
    )?;
    let got = sha256_file(out)?;
    if format!("sha256:{got}") != digest {
        return Err(io::Error::other(format!(
            "layer digest mismatch for {url}: got sha256:{got}"
        )));
    }
    Ok(())
}

/// An anonymous pull token, by the distribution-spec challenge flow: probe
/// `/v2/`, and on a Bearer challenge ask the named realm for a pull-scoped
/// token with no credentials. `None` when the registry never challenged.
fn token(image: &ImageRef) -> io::Result<Option<String>> {
    let probe = format!("https://{}/v2/", image.api_host());
    let response = curl(&probe, &[], None)?;
    if response.status != 401 {
        return Ok(None);
    }
    let challenge = response
        .headers
        .iter()
        .find_map(|(name, value)| {
            name.eq_ignore_ascii_case("www-authenticate")
                .then_some(value.as_str())
        })
        .and_then(|v| v.strip_prefix("Bearer "))
        .ok_or_else(|| io::Error::other(format!("{probe}: unsupported auth challenge")))?;
    let field = |name: &str| {
        challenge.split(',').find_map(|part| {
            part.trim()
                .strip_prefix(name)?
                .strip_prefix("=\"")?
                .strip_suffix('"')
                .map(str::to_string)
        })
    };
    let realm = field("realm")
        .ok_or_else(|| io::Error::other(format!("{probe}: auth challenge names no realm")))?;
    let mut url = format!("{realm}?scope=repository:{}:pull", image.repository);
    if let Some(service) = field("service") {
        url.push_str(&format!("&service={service}"));
    }
    let body = http_get(&url, "application/json", None, None)?;
    let json: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| io::Error::other(format!("bad token response from {url}: {e}")))?;
    let token = json["token"]
        .as_str()
        .or_else(|| json["access_token"].as_str())
        .ok_or_else(|| io::Error::other(format!("no token in response from {url}")))?;
    Ok(Some(token.to_string()))
}

/// GET `url`, failing on any non-2xx status. With `out` the body streams to
/// that file (and an empty Vec returns); otherwise the body is the result.
fn http_get(
    url: &str,
    accept: &str,
    token: Option<&str>,
    out: Option<&Path>,
) -> io::Result<Vec<u8>> {
    let mut headers = vec![format!("Accept: {accept}")];
    if let Some(token) = token {
        headers.push(format!("Authorization: Bearer {token}"));
    }
    let response = curl(url, &headers, out)?;
    if !(200..300).contains(&response.status) {
        return Err(io::Error::other(format!("{url}: HTTP {}", response.status)));
    }
    Ok(response.body)
}

/// A completed exchange: the final status, the final response's headers, and
/// the body — empty when it streamed to a file instead.
struct Response {
    status: u32,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

/// One `curl(1)` request: follows redirects (a registry hands blob GETs off
/// to a CDN; curl itself drops the Authorization header on a cross-host hop),
/// keeping the final response. The body lands in `out` when given.
fn curl(url: &str, headers: &[String], out: Option<&Path>) -> io::Result<Response> {
    let mut cmd = Command::new("curl");
    cmd.args(["-sS", "-L", "--connect-timeout", "30", "-D", "-"]);
    for header in headers {
        cmd.args(["-H", header]);
    }
    match out {
        Some(path) => cmd.arg("-o").arg(path),
        // The body follows the header dump on stdout; a blank line ends the
        // final header block, so the split below is unambiguous.
        None => cmd.arg("-o").arg("/dev/stdout"),
    };
    cmd.arg(url);
    let output = cmd.output().map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound {
            io::Error::new(e.kind(), "pulling an image needs curl(1) on PATH")
        } else {
            e
        }
    })?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "curl {url}: {}",
            String::from_utf8_lossy(&output.stderr).trim(),
        )));
    }
    // Each hop of a redirect chain dumps its own header block; the last block
    // describes the response the body came from.
    let stdout = output.stdout;
    let mut cursor = 0;
    let mut status = 0u32;
    let mut headers = Vec::new();
    while stdout[cursor..].starts_with(b"HTTP/") {
        let Some(end) = find(&stdout[cursor..], b"\r\n\r\n") else {
            break;
        };
        let block = &stdout[cursor..cursor + end];
        cursor += end + 4;
        let mut lines = block.split(|&b| b == b'\r');
        status = lines
            .next()
            .and_then(|l| {
                String::from_utf8_lossy(l)
                    .split_whitespace()
                    .nth(1)?
                    .parse()
                    .ok()
            })
            .unwrap_or(0);
        headers = block
            .split(|&b| b == b'\n')
            .filter_map(|line| {
                let line = String::from_utf8_lossy(line);
                let (name, value) = line.trim_end_matches('\r').split_once(':')?;
                Some((name.trim().to_string(), value.trim().to_string()))
            })
            .collect();
        // An informational response (100 Continue) precedes the real one.
        if (100..200).contains(&status) {
            continue;
        }
        if (300..400).contains(&status) {
            continue;
        }
        break;
    }
    Ok(Response {
        status,
        headers,
        body: stdout[cursor.min(stdout.len())..].to_vec(),
    })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn sha256(bytes: &[u8]) -> io::Result<String> {
    use std::io::Write as _;
    use std::process::Stdio;

    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                io::Error::new(e.kind(), "pulling an image needs sha256sum(1) on PATH")
            } else {
                e
            }
        })?;
    child.stdin.take().expect("piped stdin").write_all(bytes)?;
    digest_from(child.wait_with_output()?)
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let output = Command::new("sha256sum").arg(path).output()?;
    digest_from(output)
}

fn digest_from(output: std::process::Output) -> io::Result<String> {
    if !output.status.success() {
        return Err(io::Error::other("sha256sum failed"));
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
        .ok_or_else(|| io::Error::other("sha256sum produced no digest"))
}

/// Apply one layer tar onto the staged tree. Two passes, whiteouts first:
/// the OCI spec asks producers to order whiteouts ahead of their siblings
/// but consumers not to depend on it, and a second decompression is cheap
/// next to the download.
fn apply_layer(blob: &Path, media_type: &str, data: &Path) -> io::Result<()> {
    let mut skipped = 0u32;
    for entry in archive(blob, media_type)?.entries()? {
        let entry = entry?;
        match classify(&entry.path()?) {
            LayerEntry::Opaque(dir) => {
                // The directory replaces its lower content wholesale; what
                // this same layer puts back arrives in the content pass.
                let Some(dir) = staged_path(data, &dir) else {
                    skipped += 1;
                    continue;
                };
                if dir != data {
                    remove(&dir)?;
                    fs::create_dir_all(&dir)?;
                }
            }
            LayerEntry::Whiteout(victim) => {
                let Some(victim) = staged_path(data, &victim) else {
                    skipped += 1;
                    continue;
                };
                if victim != data {
                    remove(&victim)?;
                }
            }
            LayerEntry::Content => {}
        }
    }
    for entry in archive(blob, media_type)?.entries()? {
        let mut entry = entry?;
        let rel = entry.path()?.into_owned();
        if !matches!(classify(&rel), LayerEntry::Content) {
            continue;
        }
        use tar::EntryType;
        match entry.header().entry_type() {
            // Device nodes need privilege the sandbox deliberately lacks,
            // and the host's /dev serves the guest anyway.
            EntryType::Char | EntryType::Block => continue,
            _ => {}
        }
        // A later layer may change an entry's type; tar itself only
        // overwrites same-type entries, so clear the slot first. An existing
        // directory stays for a directory entry — its children are already
        // merged content.
        if let Some(target) = staged_path(data, &rel)
            && target != data
        {
            match fs::symlink_metadata(&target) {
                Ok(md) if md.is_dir() && entry.header().entry_type() == EntryType::Directory => {}
                Ok(_) => remove(&target)?,
                Err(_) => {}
            }
        }
        if !entry.unpack_in(data)? {
            skipped += 1;
        }
    }
    if skipped > 0 {
        eprintln!("chimera:   skipped {skipped} unsafe path(s) in layer");
    }
    Ok(())
}

/// The staged location of a layer-relative path, or `None` when acting on it
/// could reach outside the stage: a climbing component, or an intermediate
/// that resolves through a symlink — a layer that stages `usr` as an absolute
/// symlink and then whiteouts `usr/bin` must not delete the host's. The tar
/// crate's `unpack_in` enforces the same rule for extraction; this covers the
/// removals this module does itself.
fn staged_path(data: &Path, rel: &Path) -> Option<PathBuf> {
    use std::path::Component;

    let mut out = data.to_path_buf();
    for c in rel.components() {
        match c {
            Component::CurDir => {}
            Component::Normal(name) => {
                match fs::symlink_metadata(&out) {
                    Ok(md) if md.file_type().is_symlink() => return None,
                    _ => {}
                }
                out.push(name);
            }
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(out)
}

fn remove(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(md) if md.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

fn archive(blob: &Path, media_type: &str) -> io::Result<tar::Archive<Box<dyn Read>>> {
    let file = fs::File::open(blob)?;
    let reader: Box<dyn Read> = if media_type.contains("zstd") {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("zstd-compressed layers are not supported yet ({media_type})"),
        ));
    } else if media_type.contains("gzip") {
        Box::new(flate2::read::MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut archive = tar::Archive::new(reader);
    archive.set_preserve_permissions(true);
    archive.set_preserve_ownerships(false);
    archive.set_unpack_xattrs(false);
    Ok(archive)
}

/// What a layer tar entry means for the flattened tree, by the OCI whiteout
/// convention: `.wh..wh..opq` marks its directory opaque, any other `.wh.`
/// prefix deletes the named sibling.
enum LayerEntry {
    Opaque(PathBuf),
    Whiteout(PathBuf),
    Content,
}

fn classify(rel: &Path) -> LayerEntry {
    let Some(name) = rel.file_name().and_then(|n| n.to_str()) else {
        return LayerEntry::Content;
    };
    let parent = rel.parent().unwrap_or(Path::new(""));
    if name == ".wh..wh..opq" {
        LayerEntry::Opaque(parent.to_path_buf())
    } else if let Some(victim) = name.strip_prefix(".wh.") {
        LayerEntry::Whiteout(parent.join(victim))
    } else {
        LayerEntry::Content
    }
}

fn human_size(bytes: u64) -> String {
    match bytes {
        0..1024 => format!("{bytes}B"),
        1024..1048576 => format!("{}K", bytes / 1024),
        1048576..1073741824 => format!("{}M", bytes / 1048576),
        _ => format!("{}G", bytes / 1073741824),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(locator: &str) -> (String, String, String, String) {
        let r = ImageRef::parse(locator).unwrap();
        (r.registry, r.repository, r.reference, r.canonical)
    }

    #[test]
    fn bare_name_normalizes_to_docker_hub_library() {
        assert_eq!(
            parts("docker:debian:13-slim"),
            (
                "docker.io".into(),
                "library/debian".into(),
                "13-slim".into(),
                "docker:docker.io/library/debian:13-slim".into(),
            )
        );
    }

    #[test]
    fn missing_tag_reads_as_latest() {
        assert_eq!(parts("docker:alpine").2, "latest");
    }

    #[test]
    fn registry_host_is_recognized_by_dot() {
        assert_eq!(
            parts("docker:ghcr.io/acme/agent:v2"),
            (
                "ghcr.io".into(),
                "acme/agent".into(),
                "v2".into(),
                "docker:ghcr.io/acme/agent:v2".into(),
            )
        );
    }

    #[test]
    fn slash_name_without_host_stays_on_docker_hub() {
        assert_eq!(parts("docker:penberg/tool:1").1, "penberg/tool".to_string());
    }

    #[test]
    fn digest_pins_the_reference() {
        let digest = "sha256:38a76d01668772e381ad2826d876627c89e7133e2f8a0f5d567306798b0f2a16";
        let (_, _, reference, canonical) = parts(&format!("docker:debian:13-slim@{digest}"));
        assert_eq!(reference, digest);
        assert_eq!(
            canonical,
            format!("docker:docker.io/library/debian@{digest}")
        );
    }

    #[test]
    fn locator_without_scheme_is_refused() {
        assert!(ImageRef::parse("debian:13-slim").is_err());
    }

    #[test]
    fn whiteouts_classify_by_the_oci_convention() {
        assert!(matches!(
            classify(Path::new("usr/bin/.wh.perl")),
            LayerEntry::Whiteout(p) if p == Path::new("usr/bin/perl")
        ));
        assert!(matches!(
            classify(Path::new("etc/apt/.wh..wh..opq")),
            LayerEntry::Opaque(p) if p == Path::new("etc/apt")
        ));
        assert!(matches!(
            classify(Path::new("usr/bin/perl")),
            LayerEntry::Content
        ));
    }
}
