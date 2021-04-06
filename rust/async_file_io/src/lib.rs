#[cfg(test)]
mod tests {
    use std::io::Result;
    use std::time::Instant;
    use tempfile::NamedTempFile;

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
        let write_loop_cnt = 1000000;
        let read_loop_cnt = 100;
        let bytes_data = b"party parrot".to_vec();

        let f = NamedTempFile::new()?;
        measure!("[1W] sync write small bytes", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&f)?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data)?;
            }
        });

        measure!("[1R] sync read all bytes", {
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read(&f)?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], b'p');
                assert_eq!(buf.last(), Some(&b't'));
            }
        });

        let f = NamedTempFile::new()?;
        measure!("[1W]async write small bytes", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let mut f = BufWriter::new(File::create(&f).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data).await?;
            }
            f.flush().await?;
        });

        measure!("[1R]async read all bytes", {
            use tokio::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read(&f).await?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], b'p');
                assert_eq!(buf.last(), Some(&b't'));
            }
        });

        Ok(())
    }

    #[tokio::test]
    async fn write_big_bytes() -> Result<()> {
        let write_loop_cnt = 5;
        let read_loop_cnt = 2;
        let bytes_data: Vec<u8> = vec![0xB; 100_000_000]; // all B array

        let f = NamedTempFile::new()?;
        measure!("[2W] sync write big bytes", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&f)?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data)?;
            }
        });

        measure!("[2R] sync read all bytes", {
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read(&f)?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], 0xB);
                assert_eq!(buf.last(), Some(&0xB));
            }
        });

        let f = NamedTempFile::new()?;
        measure!("[2W]async write big bytes", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let mut f = BufWriter::new(File::create(&f).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(&bytes_data).await?;
            }
            f.flush().await?;
        });

        measure!("[2R]async read all bytes", {
            use tokio::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read(&f).await?;
                assert_eq!(buf.len(), bytes_data.len() * write_loop_cnt);
                assert_eq!(buf[0], 0xB);
                assert_eq!(buf.last(), Some(&0xB));
            }
        });

        Ok(())
    }

    #[tokio::test]
    async fn write_small_str() -> Result<()> {
        let write_loop_cnt = 1000000;
        let read_loop_cnt = 100;
        let str_data = "party parrot".to_string();

        let f = NamedTempFile::new()?;
        measure!("[3W] sync write small str", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&f)?);
            for _ in 0..write_loop_cnt {
                write!(f, "{}", &str_data)?;
            }
        });

        measure!("[3R] sync read all string", {
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&f)?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('p'));
                assert_eq!(buf.chars().last(), Some('t'));
            }
        });

        let f = NamedTempFile::new()?;
        measure!("[3W] sync write small str (use write)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&f)?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes())?;
            }
        });

        {
            // just verify
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&f)?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('p'));
                assert_eq!(buf.chars().last(), Some('t'));
            }
        }

        let f = NamedTempFile::new()?;
        measure!("[3W]async write small str", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let mut f = BufWriter::new(File::create(&f).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes()).await?;
            }
            f.flush().await?;
        });

        measure!("[3R]async read all string", {
            use tokio::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&f).await?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('p'));
                assert_eq!(buf.chars().last(), Some('t'));
            }
        });

        Ok(())
    }

    #[tokio::test]
    async fn write_big_str() -> Result<()> {
        let write_loop_cnt = 10;
        let read_loop_cnt = 1;
        let str_data = "a".repeat(10_000_000);

        let f = NamedTempFile::new()?;
        measure!("[4W] sync write big str (use macro)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&f)?);
            for _ in 0..write_loop_cnt {
                write!(f, "{}", &str_data)?;
            }
        });

        measure!("[3R] sync read all string", {
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&f)?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('a'));
                assert_eq!(buf.chars().last(), Some('a'));
            }
        });

        let f = NamedTempFile::new()?;
        measure!("[4W] sync write big str (use write)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let mut f = BufWriter::new(File::create(&f)?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes())?;
            }
        });

        {
            // just verify
            use std::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&f)?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('a'));
                assert_eq!(buf.chars().last(), Some('a'));
            }
        }

        let f = NamedTempFile::new()?;
        measure!("[4W]async write big str", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let mut f = BufWriter::new(File::create(&f).await?);
            for _ in 0..write_loop_cnt {
                f.write_all(str_data.as_bytes()).await?;
            }
            f.flush().await?;
        });

        measure!("[4R]async read all string", {
            use tokio::fs;

            for _ in 0..read_loop_cnt {
                let buf = fs::read_to_string(&f).await?;
                assert_eq!(buf.len(), str_data.len() * write_loop_cnt);
                assert_eq!(buf.chars().nth(0), Some('a'));
                assert_eq!(buf.chars().last(), Some('a'));
            }
        });

        Ok(())
    }

    #[tokio::test]
    async fn write_read_lines() -> Result<()> {
        let line_num = 1000000;
        let str_data = "party parrot\n".to_string();

        {
            let f = NamedTempFile::new()?;

            measure!("[5W] sync write lines", {
                use std::fs::File;
                use std::io::BufWriter;
                use std::io::Write;

                let mut f = BufWriter::new(File::create(&f)?);
                for _ in 0..line_num {
                    f.write_all(str_data.as_bytes())?;
                }
            });

            measure!("[5R] sync read lines", {
                use std::fs::File;
                use std::io::BufRead;
                use std::io::BufReader;

                let f = BufReader::new(File::open(f)?);
                assert_eq!(f.lines().count(), line_num);
            });
        }

        {
            let f = NamedTempFile::new()?;

            measure!("[5W]async write lines", {
                use tokio::fs::File;
                use tokio::io::AsyncWriteExt;
                use tokio::io::BufWriter;

                let mut f = BufWriter::new(File::create(&f).await?);
                for _ in 0..line_num {
                    f.write_all(str_data.as_bytes()).await?;
                }
                f.flush().await?;
            });

            measure!("[5R]async read lines", {
                use tokio::fs::File;
                use tokio::io::AsyncBufReadExt;
                use tokio::io::BufReader;

                let f = BufReader::new(File::open(f).await?);
                let mut lines = f.lines();
                let mut count = 0;
                while let Some(_) = lines.next_line().await? {
                    count += 1;
                }

                assert_eq!(count, line_num);
            });
        }

        Ok(())
    }
}
