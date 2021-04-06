pub trait Command {
    fn new() -> Self;
    fn execute(&mut self, file_name: &str);
    fn undo(&mut self);
}

pub struct HideFileCommand {
    _hidden_files: Vec<String>,
}

impl Command for HideFileCommand {
    fn new() -> Self {
        HideFileCommand {
            _hidden_files: Vec::new(),
        }
    }
    fn execute(&mut self, file_name: &str) {
        println!("hiding {}", file_name);
        self._hidden_files.push(file_name.to_string())
    }
    fn undo(&mut self) {
        let filename = self._hidden_files.pop().unwrap();
        println!("un-hiding {}", filename);
    }
}

pub struct DeleteFileCommand {
    _deleted_files: Vec<String>,
}

impl Command for DeleteFileCommand {
    fn new() -> Self {
        DeleteFileCommand {
            _deleted_files: Vec::new(),
        }
    }
    fn execute(&mut self, file_name: &str) {
        println!("deleting {}", file_name);
        self._deleted_files.push(file_name.to_string())
    }
    fn undo(&mut self) {
        let filename = self._deleted_files.pop().unwrap();
        println!("restoring {}", filename);
    }
}

pub struct MenuItem<T> {
    _command: T,
}

impl<T: Command> MenuItem<T> {
    pub fn new() -> Self {
        MenuItem { _command: T::new() }
    }
    pub fn on_do_press(&mut self, file_name: &str) {
        self._command.execute(file_name);
    }
    pub fn on_undo_press(&mut self) {
        self._command.undo();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commander() {
        let mut item1: MenuItem<HideFileCommand> = MenuItem::new();
        let mut item2: MenuItem<DeleteFileCommand> = MenuItem::new();
        let test_file_name = "test-file";
        item1.on_do_press(&test_file_name);
        item1.on_undo_press();
        item2.on_do_press(&test_file_name);
        item2.on_undo_press();
    }
}
