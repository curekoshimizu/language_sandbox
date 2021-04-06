use std::fs;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Read;
use std::io::Result;
use std::io::Write;
use std::path::PathBuf;

// bad
pub fn write_bytes(loop_cnt: i32, path: &PathBuf, data: &Vec<u8>) -> Result<()> {
    let mut f = File::create(path)?;
    for _ in 0..loop_cnt {
        f.write_all(data)?;
    }
    Ok(())
}

// good!
pub fn buffered_write_bytes(loop_cnt: i32, path: &PathBuf, data: &Vec<u8>) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    for _ in 0..loop_cnt {
        f.write_all(data)?;
    }
    Ok(())
}

// bad
pub fn write_string_macro(loop_cnt: i32, path: &PathBuf, data: &String) -> Result<()> {
    let mut f = File::create(path)?;
    for _ in 0..loop_cnt {
        write!(f, "{}", data)?;
    }
    Ok(())
}

// bad
pub fn write_string(loop_cnt: i32, path: &PathBuf, data: &String) -> Result<()> {
    let mut f = File::create(path)?;
    for _ in 0..loop_cnt {
        f.write_all(data.as_bytes())?;
    }
    Ok(())
}

// good!
pub fn buffered_write_string(loop_cnt: i32, path: &PathBuf, data: &String) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    for _ in 0..loop_cnt {
        write!(f, "{}", data)?;
    }
    Ok(())
}

// not simple
pub fn read_all_bytes(loop_cnt: i32, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let mut f = File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        assert_eq!(buf.len(), 100000000);
    }

    Ok(())
}

// not simple
pub fn buffered_read_all_bytes(loop_cnt: i32, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let mut f = BufReader::new(File::open(path)?);
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        assert_eq!(buf.len(), 100000000);
    }

    Ok(())
}

// not simple
pub fn fs_read_all_bytes(loop_cnt: i32, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let buf = fs::read(&path)?;
        assert_eq!(buf.len(), 100000000);
    }

    Ok(())
}

// not simple
pub fn read_all_strings(loop_cnt: i32, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let mut f = File::open(path)?;
        let mut buf = String::new();
        f.read_to_string(&mut buf)?;
        assert_eq!(buf.len(), 1300000);
    }

    Ok(())
}

// not simple
pub fn buffered_read_all_strings(loop_cnt: i32, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let mut f = BufReader::new(File::open(path)?);
        let mut buf = String::new();
        f.read_to_string(&mut buf)?;
        assert_eq!(buf.len(), 1300000);
    }

    Ok(())
}

// simple!!
pub fn fs_read_all_strings(loop_cnt: i32, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let buf = fs::read_to_string(path)?;
        assert_eq!(buf.len(), 1300000);
    }

    Ok(())
}

pub fn read_lines(loop_cnt: i32, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let f = BufReader::new(File::open(path)?);
        assert_eq!(f.lines().count(), 100000);
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
    fn test_write_file() -> Result<()> {
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
            measure!("buffered write byte★", {
                buffered_write_bytes(
                    loop_cnt,
                    &temp_dir.join("small_buffered_write_bytes.txt"),
                    &bytes_data,
                )?;
            }); // best!
        }

        {
            println!("[big data]");
            let loop_cnt = 8;
            let bytes_data: Vec<u8> = vec![0xB; 100_000_000]; // all B array
            measure!("write byte", {
                write_bytes(loop_cnt, &temp_dir.join("big_write_bytes.txt"), &bytes_data)?;
            });
            measure!("buffered write byte★", {
                buffered_write_bytes(
                    loop_cnt,
                    &temp_dir.join("big_buffered_write_bytes.txt"),
                    &bytes_data,
                )?;
            }); // best!
        }

        {
            println!("[small data]");
            let loop_cnt = 100000;
            let str_data = "party parrot".to_string();
            measure!("write string", {
                write_string_macro(
                    loop_cnt,
                    &temp_dir.join("small_write_string.txt"),
                    &str_data,
                )?;
            });
            measure!("buffered write string★", {
                buffered_write_string(
                    loop_cnt,
                    &temp_dir.join("small_buffered_write_string.txt"),
                    &str_data,
                )?;
            }); // best!
        }

        {
            println!("[big data]");
            let loop_cnt = 10;
            let str_data = "a".repeat(10_000_000);
            measure!("write string macro", {
                write_string_macro(
                    loop_cnt,
                    &temp_dir.join("big_write_string_macro_1.txt"),
                    &str_data,
                )?;
            });
            measure!("write string", {
                write_string(
                    loop_cnt,
                    &temp_dir.join("big_write_string_1.txt"),
                    &str_data,
                )?;
            });
            measure!("buffered write string★", {
                buffered_write_string(
                    loop_cnt,
                    &temp_dir.join("big_buffered_write_string_1.txt"),
                    &str_data,
                )?;
            }); // best!
        }

        {
            println!("[big data]");
            let loop_cnt = 10;
            let str_data = "a".repeat(10_000_000);
            measure!("write string macro", {
                write_string_macro(
                    loop_cnt,
                    &temp_dir.join("big_write_string_macro_2.txt"),
                    &str_data,
                )?;
            });
            measure!("write string", {
                write_string(
                    loop_cnt,
                    &temp_dir.join("big_write_string_2.txt"),
                    &str_data,
                )?;
            });
            measure!("buffered write string★", {
                buffered_write_string(
                    loop_cnt,
                    &temp_dir.join("big_buffered_write_string_2.txt"),
                    &str_data,
                )?;
            }); // best!
        }

        Ok(())
    }

    #[test]
    fn test_read_file() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        {
            println!("[read all as bytes]");
            let bytes_data: Vec<u8> = vec![0xB; 100_000_000]; // all B array
            let read_file_bytes = temp_dir.join("data");
            buffered_write_bytes(1, &read_file_bytes, &bytes_data)?;

            measure!("read all as bytes", read_all_bytes(10, &read_file_bytes)?);
            measure!(
                "buffered read all as bytes",
                buffered_read_all_bytes(10, &read_file_bytes)?
            );
            measure!(
                "fs read all as bytes★",
                fs_read_all_bytes(10, &read_file_bytes)?
            ); // best!
        }

        let str_data = "party parrot\n".to_string();
        let read_file_string = temp_dir.join("string");
        buffered_write_string(100000, &&read_file_string, &str_data)?;
        {
            println!("[read all as string]");

            measure!(
                "read all as string",
                read_all_strings(1000, &read_file_string)?
            );
            measure!(
                "buffered read all as string",
                buffered_read_all_strings(1000, &read_file_string)?
            );
            measure!(
                "fs read all as string★",
                fs_read_all_strings(1000, &read_file_string)?
            ); // best!
        }

        {
            println!("[read lines]");
            measure!("read lines★", read_lines(10, &read_file_string)?);
        }

        Ok(())
    }
}
