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

        let loop_cnt = 1000000;
        let bytes_data = b"party parrot".to_vec();

        measure!(" sync write small bytes", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let fname = temp_dir.join("write_small_bytes_sync");
            let mut f = BufWriter::new(File::create(fname)?);
            for _ in 0..loop_cnt {
                f.write(&bytes_data)?;
            }
        });

        measure!("async write small bytes", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let fname = temp_dir.join("write_small_bytes_async");
            let mut f = BufWriter::new(File::create(fname).await?);
            for _ in 0..loop_cnt {
                f.write(&bytes_data).await?;
            }
        });

        // TODO: add assert

        Ok(())
    }

    #[tokio::test]
    async fn write_big_bytes() -> Result<(), Error> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let loop_cnt = 10;
        let bytes_data: Vec<u8> = vec![0xB; 100_000_000]; // all B array

        measure!(" sync write big bytes", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let fname = temp_dir.join("write_big_bytes_sync");
            let mut f = BufWriter::new(File::create(fname)?);
            for _ in 0..loop_cnt {
                f.write(&bytes_data)?;
            }
        });

        measure!("async write big bytes", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let fname = temp_dir.join("write_big_bytes_async");
            let mut f = BufWriter::new(File::create(fname).await?);
            for _ in 0..loop_cnt {
                f.write(&bytes_data).await?;
            }
        });

        // TODO: add assert

        Ok(())
    }

    #[tokio::test]
    async fn write_small_str() -> Result<(), Error> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let loop_cnt = 100000;
        let str_data = "party parrot".to_string();

        measure!(" sync write small str (use macro)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let fname = temp_dir.join("write_small_str_sync_use_macro");
            let mut f = BufWriter::new(File::create(fname)?);
            for _ in 0..loop_cnt {
                write!(f, "{}", &str_data)?;
            }
        });

        measure!(" sync write small str (use write)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let fname = temp_dir.join("write_small_str_sync_use_write");
            let mut f = BufWriter::new(File::create(fname)?);
            for _ in 0..loop_cnt {
                f.write(str_data.as_bytes())?;
            }
        });

        measure!("async write small str", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let fname = temp_dir.join("write_small_str_async");
            let mut f = BufWriter::new(File::create(fname).await?);
            for _ in 0..loop_cnt {
                f.write(str_data.as_bytes()).await?;
            }
        });

        Ok(())
    }

    #[tokio::test]
    async fn write_big_str() -> Result<(), Error> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let loop_cnt = 10;
        let str_data = "a".repeat(10_000_000);

        measure!(" sync write big str (use macro)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let fname = temp_dir.join("write_big_str_sync_use_macro");
            let mut f = BufWriter::new(File::create(fname)?);
            for _ in 0..loop_cnt {
                write!(f, "{}", &str_data)?;
            }
        });

        measure!(" sync write big str (use write)", {
            use std::fs::File;
            use std::io::BufWriter;
            use std::io::Write;

            let fname = temp_dir.join("write_big_str_sync_use_write");
            let mut f = BufWriter::new(File::create(fname)?);
            for _ in 0..loop_cnt {
                f.write(str_data.as_bytes())?;
            }
        });

        measure!("async write big str", {
            use tokio::fs::File;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufWriter;

            let fname = temp_dir.join("write_big_str_async");
            let mut f = BufWriter::new(File::create(fname).await?);
            for _ in 0..loop_cnt {
                f.write(str_data.as_bytes()).await?;
            }
        });

        Ok(())
    }
}
