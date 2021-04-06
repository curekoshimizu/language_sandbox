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
pub fn write_bytes(loop_cnt: usize, path: &PathBuf, data: &Vec<u8>) -> Result<()> {
    let mut f = File::create(path)?;
    for _ in 0..loop_cnt {
        f.write_all(data)?;
    }
    Ok(())
}

// good!
pub fn buffered_write_bytes(loop_cnt: usize, path: &PathBuf, data: &Vec<u8>) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    for _ in 0..loop_cnt {
        f.write_all(data)?;
    }
    Ok(())
}

// bad
pub fn write_string_macro(loop_cnt: usize, path: &PathBuf, data: &String) -> Result<()> {
    let mut f = File::create(path)?;
    for _ in 0..loop_cnt {
        write!(f, "{}", data)?;
    }
    Ok(())
}

// bad
pub fn write_string(loop_cnt: usize, path: &PathBuf, data: &String) -> Result<()> {
    let mut f = File::create(path)?;
    for _ in 0..loop_cnt {
        f.write_all(data.as_bytes())?;
    }
    Ok(())
}

// good!
pub fn buffered_write_string(loop_cnt: usize, path: &PathBuf, data: &String) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    for _ in 0..loop_cnt {
        write!(f, "{}", data)?;
    }
    Ok(())
}

// not simple
pub fn read_all_bytes(loop_cnt: usize, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let mut f = File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        assert_eq!(buf.len(), 100000000);
    }

    Ok(())
}

// not simple
pub fn buffered_read_all_bytes(loop_cnt: usize, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let mut f = BufReader::new(File::open(path)?);
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        assert_eq!(buf.len(), 100000000);
    }

    Ok(())
}

// not simple
pub fn fs_read_all_bytes(loop_cnt: usize, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let buf = fs::read(&path)?;
        assert_eq!(buf.len(), 100000000);
    }

    Ok(())
}

// not simple
pub fn read_all_strings(loop_cnt: usize, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let mut f = File::open(path)?;
        let mut buf = String::new();
        f.read_to_string(&mut buf)?;
        assert_eq!(buf.len(), 1300000);
    }

    Ok(())
}

// not simple
pub fn buffered_read_all_strings(loop_cnt: usize, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let mut f = BufReader::new(File::open(path)?);
        let mut buf = String::new();
        f.read_to_string(&mut buf)?;
        assert_eq!(buf.len(), 1300000);
    }

    Ok(())
}

// simple!!
pub fn fs_read_all_strings(loop_cnt: usize, path: &PathBuf) -> Result<()> {
    for _ in 0..loop_cnt {
        let buf = fs::read_to_string(path)?;
        assert_eq!(buf.len(), 1300000);
    }

    Ok(())
}

pub fn read_lines(line_num: usize, path: &PathBuf) -> Result<()> {
    let f = BufReader::new(File::open(path)?);
    assert_eq!(f.lines().count(), line_num);

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
    fn write_small_bytes() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        {
            let loop_cnt = 100000;
            let bytes_data = b"party parrot".to_vec();
            measure!("[1] write byte", {
                write_bytes(loop_cnt, &temp_dir.join("a"), &bytes_data)?;
            });
            measure!("[1] buffered write byte★", {
                buffered_write_bytes(loop_cnt, &temp_dir.join("b"), &bytes_data)?;
            }); // best!
        }

        Ok(())
    }

    #[test]
    fn write_big_bytes() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        {
            let loop_cnt = 8;
            let bytes_data: Vec<u8> = vec![0xB; 100_000_000]; // all B array
            measure!("[2] write byte", {
                write_bytes(loop_cnt, &temp_dir.join("a"), &bytes_data)?;
            });
            measure!("[2] buffered write byte★", {
                buffered_write_bytes(loop_cnt, &temp_dir.join("b"), &bytes_data)?;
            }); // best!
        }

        Ok(())
    }

    #[test]
    fn write_small_str() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        {
            let loop_cnt = 100000;
            let str_data = "party parrot".to_string();
            measure!("[3] write string macro", {
                write_string_macro(loop_cnt, &temp_dir.join("a"), &str_data)?;
            });
            measure!("[3] write string", {
                write_string(loop_cnt, &temp_dir.join("b"), &str_data)?;
            });
            measure!("[3] buffered write string★", {
                buffered_write_string(loop_cnt, &temp_dir.join("c"), &str_data)?;
            }); // best!
        }

        Ok(())
    }

    #[test]
    fn write_big_str() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        {
            let loop_cnt = 10;
            let str_data = "a".repeat(20_000_000);
            measure!("[4] write string macro", {
                write_string_macro(loop_cnt, &temp_dir.join("a"), &str_data)?;
            });
            measure!("[4] write string", {
                write_string(loop_cnt, &temp_dir.join("b"), &str_data)?;
            });
            measure!("[4] buffered write string★", {
                buffered_write_string(loop_cnt, &temp_dir.join("c"), &str_data)?;
            }); // best!
        }

        Ok(())
    }

    #[test]
    fn read_bytes() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        {
            let bytes_data: Vec<u8> = vec![0xB; 100_000_000]; // all B array
            let read_file_bytes = temp_dir.join("data");
            buffered_write_bytes(1, &read_file_bytes, &bytes_data)?;

            measure!(
                "[5] read all as bytes",
                read_all_bytes(10, &read_file_bytes)?
            );
            measure!(
                "[5] buffered read all as bytes",
                buffered_read_all_bytes(10, &read_file_bytes)?
            );
            measure!(
                "[5] fs read all as bytes★",
                fs_read_all_bytes(10, &read_file_bytes)?
            ); // best!
        }

        Ok(())
    }

    #[test]
    fn read_str() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let str_data = "party parrot\n".to_string();
        let read_file_string = temp_dir.join("string");
        buffered_write_string(100000, &&read_file_string, &str_data)?;
        {
            measure!(
                "[6] read all as string",
                read_all_strings(1000, &read_file_string)?
            );
            measure!(
                "[6] buffered read all as string",
                buffered_read_all_strings(1000, &read_file_string)?
            );
            measure!(
                "[6] fs read all as string★",
                fs_read_all_strings(1000, &read_file_string)?
            ); // best!
        }

        Ok(())
    }

    #[test]
    fn write_read_lines() -> Result<()> {
        let tmp = TempDir::new()?;
        let temp_dir = tmp.path().to_path_buf();

        let line_num = 1000000;
        let str_data = "party parrot\n".to_string();
        let fname = temp_dir.join("a");

        measure!("[7] write lines", {
            buffered_write_string(line_num, &fname, &str_data)?;
        });

        measure!("[7] read lines", read_lines(line_num, &fname)?);

        Ok(())
    }
}
