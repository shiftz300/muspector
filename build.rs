fn main() {
    println!("cargo:rerun-if-changed=assets/AppIcon.ico");

    #[cfg(windows)]
    {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("assets/AppIcon.ico");
        resource.set("FileDescription", "Muspector audio inspector");
        resource.set("ProductName", "Muspector");
        resource.set("OriginalFilename", "muspector.exe");
        resource
            .compile()
            .expect("failed to embed Windows resources");
    }
}
