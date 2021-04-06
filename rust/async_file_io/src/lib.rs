#[cfg(test)]
mod tests {
    use std::io::Result;
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
    async fn write_small_bytes() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let write_loop_cnt = 1000000;
        let read_loop_cnt = 100;
        let bytes_data = b"party parrot".to_vec();

        let fname = temp_dir.join("write_small_bytes_sync");
        measure!("[1W] sync write small bytes", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&fname)?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data)?;
            }
        });

        measure!("[1R] sync read all bytes", {
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read(&fname)?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], b'p');
                assert_eq!(buf.last(), Some(&b't'));
            }
        });

        let fname = temp_dir.join("write_small_bytes_async");
        measure!("[1W]async write small bytes", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let mut f = BufWriter::new(File::create(&fname).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data).await?;
            }
            f.flush().await?;
        });

        measure!("[1R]async read all bytes", {
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
    async fn write_big_bytes() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let write_loop_cnt = 5;
        let read_loop_cnt = 2;
        let bytes_data: Vec<u8> = vec![0xB; 100_000_000]; // all B array

        let fname = temp_dir.join("write_big_bytes_sync");
        measure!("[2W] sync write big bytes", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&fname)?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data)?;
            }
        });

        measure!("[2R] sync read all bytes", {
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read(&fname)?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], 0xB);
                assert_eq!(buf.last(), Some(&0xB));
            }
        });

        let fname = temp_dir.join("write_big_bytes_async");
        measure!("[2W]async write big bytes", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let mut f = BufWriter::new(File::create(&fname).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data).await?;
            }
            f.flush().await?;
        });

        measure!("[2R]async read all bytes", {
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
    async fn write_small_str() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let write_loop_cnt = 1000000;
        let read_loop_cnt = 100;
        let str_data = "party parrot".to_string();

        let fname = temp_dir.join("write_small_str_sync_use_macro");
        measure!("[3W] sync write small str", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&fname)?);
            for _ in 0..write_loop_cnt {
                write!(f, "{}", &str_data)?;
            }
        });

        measure!("[3R] sync read all string", {
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&fname)?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('p'));
                assert_eq!(buf.chars().last(), Some('t'));
            }
        });

        let fname = temp_dir.join("write_small_str_sync_use_write");
        measure!("[3W] sync write small str (use write)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&fname)?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes())?;
            }
        });

        {
            // just verify
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&fname)?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('p'));
                assert_eq!(buf.chars().last(), Some('t'));
            }
        }

        let fname = temp_dir.join("write_small_str_async");
        measure!("[3W]async write small str", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let mut f = BufWriter::new(File::create(&fname).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes()).await?;
            }
            f.flush().await?;
        });

        measure!("[3R]async read all string", {
            use tokio::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&fname).await?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('p'));
                assert_eq!(buf.chars().last(), Some('t'));
            }
        });

        Ok(())
    }

    #[tokio::test]
    async fn write_big_str() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let write_loop_cnt = 10;
        let read_loop_cnt = 1;
        let str_data = "a".repeat(10_000_000);

        let fname = temp_dir.join("write_big_str_sync_use_macro");
        measure!("[4W] sync write big str (use macro)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&fname)?);
            for _ in 0..write_loop_cnt {
                write!(f, "{}", &str_data)?;
            }
        });

        measure!("[3R] sync read all string", {
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&fname)?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('a'));
                assert_eq!(buf.chars().last(), Some('a'));
            }
        });

        let fname = temp_dir.join("write_big_str_sync_use_write");
        measure!("[4W] sync write big str (use write)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&fname)?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes())?;
            }
        });

        {
            // just verify
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&fname)?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('a'));
                assert_eq!(buf.chars().last(), Some('a'));
            }
        }

        let fname = temp_dir.join("write_big_str_async");
        measure!("[4W]async write big str", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let mut f = BufWriter::new(File::create(&fname).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes()).await?;
            }
            f.flush().await?;
        });

        measure!("[3R]async read all string", {
            use tokio::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&fname).await?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('a'));
                assert_eq!(buf.chars().last(), Some('a'));
            }
        });

        Ok(())
    }
}
