#[cfg(test)]
mod tests {

    use std::fs;
    use std::fs::File;
    use std::io::{BufRead, BufReader, Error, Write};
    use std::path::PathBuf;
    use std::thread;
    use tempfile::TempDir;

    #[test]
    fn temp_dir_test() -> Result<(), Error> {
        let path;
        {
            let tmp = TempDir::new()?;
            path = tmp.path().to_path_buf();
            assert!(path.exists());
        }
        // tmp dir was removed automatically!
        assert!(!path.exists());

        Ok(())
    }

    #[test]
    fn wrong_used_example() -> Result<(), Error> {
        // NOTE:
        // TempDir.path becomes None after into_path is called.
        // That's why drop won't remove temporary directory.

        let path;
        {
            let tmp = TempDir::new()?;
            path = tmp.into_path();
            assert!(path.exists());
        }
        // tmp dir was not removed automatically!!!!
        assert!(path.exists());
        fs::remove_dir_all(&path)?;
        assert!(!path.exists());

        Ok(())
    }

    #[test]
    fn pathlib_test() -> Result<(), Error> {
        let new_file: PathBuf;
        {
            // create temp dir
            let tmp_dir = TempDir::new()?;
            let tmp_path = tmp_dir.path().to_path_buf();
            assert!(tmp_path.exists());
            assert!(tmp_path.is_dir());

            // create sub dir
            let new_dir = tmp_path.join("abc");
            fs::create_dir(&new_dir).unwrap();

            // make simple file
            let file = new_dir.join("rust.txt");
            {
                let mut output = File::create(&file)?;
                write!(output, "Rust\n💖\nFun")?;
            }
            assert!(file.exists());
            assert!(file.is_file());

            // read test
            {
                let f = File::open(&file)?;
                let f = BufReader::new(f);
                assert_eq!(f.lines().count(), 3);
            }

            // copy
            new_file = tmp_path.join("new_file.txt");
            fs::copy(&file, &new_file)?;

            // remove sub dir
            fs::remove_dir_all(new_dir)?;

            // read test again
            {
                assert!(!file.exists());
                assert!(File::open(&file).is_err());
            }

            assert!(new_file.exists());
        }
        assert!(!new_file.exists());

        Ok(())
    }
}
