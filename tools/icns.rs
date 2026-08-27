use std::{env, fs, io, path::Path};

fn main() -> io::Result<()> {
    let mut args = env::args_os().skip(1);
    let output = args.next().expect("usage: icns OUTPUT PNG...");
    let kinds = [b"icp4", b"icp5", b"icp6", b"ic07", b"ic08", b"ic09", b"ic10"];
    let images: Vec<_> = args.map(fs::read).collect::<io::Result<_>>()?;
    assert_eq!(images.len(), kinds.len(), "expected seven PNG sizes");

    let length = 8 + images.iter().map(|image| 8 + image.len()).sum::<usize>();
    let mut data = Vec::with_capacity(length);
    data.extend_from_slice(b"icns");
    data.extend_from_slice(&(length as u32).to_be_bytes());
    for (kind, image) in kinds.iter().zip(images) {
        data.extend_from_slice(*kind);
        data.extend_from_slice(&((image.len() + 8) as u32).to_be_bytes());
        data.extend_from_slice(&image);
    }
    fs::write(Path::new(&output), data)
}
