#[cfg(test)]
mod tests {
    use std::io::Error;
    use std::time::Instant;
    use tempfile::TempDir;

    macro_rules! measure {
        ($name:expr, $x:expr) => {{
            let start = Instant::now();
            $x;
            let end = start.elapsed();
            println!(
                "elapsed time : {}.{:03} sec ({})",
                end.as_secs(),
                end.subsec_nanos() / 1_000_000,
                $name,
            );
        }};
    }

    #[tokio::test]
    async fn write_small_bytes() -> Result<(), Error> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let write_loop_cnt = 1000000;
        let read_loop_cnt = 1000;
        let bytes_data = b"party parrot".to_vec();

        let fname = temp_dir.join("write_small_bytes_sync");
        measure!(" sync write small bytes", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&fname)?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data)?;
            }
        });

        measure!(" sync read all bytes (1)", {
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read(&fname)?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], b'p');
                assert_eq!(buf.last(), Some(&b't'));
            }
        });

        let fname = temp_dir.join("write_small_bytes_async");
        measure!("async write small bytes", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let mut f = BufWriter::new(File::create(&fname).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data).await?;
            }
            f.flush().await?; // If it is removed, test won't pass...
        });

        measure!("async read all bytes (1)", {
            use tokio::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read(&fname).await?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], b'p');
                assert_eq!(buf.last(), Some(&b't'));
            }
        });

        Ok(())
    }

    #[tokio::test]
    async fn write_big_bytes() -> Result<(), Error> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let write_loop_cnt = 15;
        let read_loop_cnt = 1;
        let bytes_data: Vec<u8> = vec![0xB; 100_000_000]; // all B array

        let fname = temp_dir.join("write_big_bytes_sync");
        measure!(" sync write big bytes", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&fname)?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data)?;
            }
        });

        measure!(" sync read all bytes (2)", {
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read(&fname)?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], 0xB);
                assert_eq!(buf.last(), Some(&0xB));
            }
        });

        let fname = temp_dir.join("write_big_bytes_async");
        measure!("async write big bytes", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let mut f = BufWriter::new(File::create(&fname).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data).await?;
            }
            f.flush().await?; // If it is removed, test won't pass...
        });

        measure!("async read all bytes (2)", {
            use tokio::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read(&fname).await?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], 0xB);
                assert_eq!(buf.last(), Some(&0xB));
            }
        });

        Ok(())
    }

    #[tokio::test]
    async fn write_small_str() -> Result<(), Error> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let write_loop_cnt = 1000000;
        let str_data = "party parrot".to_string();

        measure!(" sync write small str (use macro)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let fname = temp_dir.join("write_small_str_sync_use_macro");
            let mut f = BufWriter::new(File::create(fname)?);
            for _ in 0..write_loop_cnt {
                write!(f, "{}", &str_data)?;
            }
        });

        measure!(" sync write small str (use write)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let fname = temp_dir.join("write_small_str_sync_use_write");
            let mut f = BufWriter::new(File::create(fname)?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes())?;
            }
        });

        measure!("async write small str", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let fname = temp_dir.join("write_small_str_async");
            let mut f = BufWriter::new(File::create(fname).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes()).await?;
            }
        });

        Ok(())
    }

    #[tokio::test]
    async fn write_big_str() -> Result<(), Error> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let write_loop_cnt = 10;
        let str_data = "a".repeat(10_000_000);

        measure!(" sync write big str (use macro)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let fname = temp_dir.join("write_big_str_sync_use_macro");
            let mut f = BufWriter::new(File::create(fname)?);
            for _ in 0..write_loop_cnt {
                write!(f, "{}", &str_data)?;
            }
        });

        measure!(" sync write big str (use write)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let fname = temp_dir.join("write_big_str_sync_use_write");
            let mut f = BufWriter::new(File::create(fname)?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes())?;
            }
        });

        measure!("async write big str", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let fname = temp_dir.join("write_big_str_async");
            let mut f = BufWriter::new(File::create(fname).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes()).await?;
            }
        });

        Ok(())
    }

    #[tokio::test]
    async fn read_all_bytes() -> Result<(), Error> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let write_loop_cnt = 20000;
        let bytes_data = b"party parrot".to_vec();

        let fname = temp_dir.join("data");
        {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&fname)?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data)?;
            }
        }

        measure!(" sync read all bytes", {
            use std::fs;

            for _ in 0..write_loop_cnt {
                let buf = fs::read(&fname)?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], b'p');
                assert_eq!(buf.last(), Some(&b't'));
            }
        });

        Ok(())
    }
}
