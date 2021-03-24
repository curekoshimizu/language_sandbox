use std::fs::File;
use std::io::BufWriter;
use std::io::Error;
use std::io::Write;
use std::path::PathBuf;

pub fn write_bytes(loop_cnt: i32, path: &PathBuf, data: &Vec<u8>) -> Result<(), Error> {
    let mut f = File::create(path)?;
    for _ in 0..loop_cnt {
        f.write(data)?;
    }
    Ok(())
}

pub fn buffered_write_bytes(loop_cnt: i32, path: &PathBuf, data: &Vec<u8>) -> Result<(), Error> {
    let mut f = BufWriter::new(File::create(path)?);
    for _ in 0..loop_cnt {
        f.write(data)?;
    }
    Ok(())
}

pub fn write_string(loop_cnt: i32, path: &PathBuf, data: &String) -> Result<(), Error> {
    let mut f = File::create(path)?;
    for _ in 0..loop_cnt {
        write!(f, "{}", data)?;
    }
    Ok(())
}

pub fn buffered_write_string(loop_cnt: i32, path: &PathBuf, data: &String) -> Result<(), Error> {
    let f = File::create(path)?;
    let mut f = BufWriter::new(f);
    for _ in 0..loop_cnt {
        write!(f, "{}", data)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_write_file() -> Result<(), Error> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        {
            println!("[small data]");
            let loop_cnt = 100000;
            let bytes_data = b"party parrot".to_vec();
            measure!("write byte", {
                write_bytes(
                    loop_cnt,
                    &temp_dir.join("small_write_bytes.txt"),
                    &bytes_data,
                )?;
            });
            measure!("buffered write byte", {
                buffered_write_bytes(
                    loop_cnt,
                    &temp_dir.join("small_buffered_write_bytes.txt"),
                    &bytes_data,
                )?;
            });
        }

        {
            println!("[big data]");
            let loop_cnt = 8;
            let bytes_data: Vec<u8> = vec![0xB; 100_000_000]; // all B array
            measure!("write byte", {
                write_bytes(loop_cnt, &temp_dir.join("big_write_bytes.txt"), &bytes_data)?;
            });
            measure!("buffered write byte", {
                buffered_write_bytes(
                    loop_cnt,
                    &temp_dir.join("big_buffered_write_bytes.txt"),
                    &bytes_data,
                )?;
            });
        }

        {
            println!("[small data]");
            let loop_cnt = 100000;
            let str_data = "party parrot".to_string();
            measure!("write string", {
                write_string(
                    loop_cnt,
                    &temp_dir.join("small_write_string.txt"),
                    &str_data,
                )?;
            });
            measure!("buffered write string", {
                buffered_write_string(
                    loop_cnt,
                    &temp_dir.join("small_buffered_write_string.txt"),
                    &str_data,
                )?;
            });
        }

        {
            println!("[big data]");
            let loop_cnt = 10;
            let str_data = "a".repeat(10_000_000);
            measure!("write string", {
                write_string(loop_cnt, &temp_dir.join("big_write_string.txt"), &str_data)?;
            });
            measure!("buffered write string", {
                buffered_write_string(
                    loop_cnt,
                    &temp_dir.join("big_buffered_write_string.txt"),
                    &str_data,
                )?;
            });
        }

        Ok(())
    }
}
