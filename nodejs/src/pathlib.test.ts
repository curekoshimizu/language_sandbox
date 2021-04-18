import fs from 'fs';
import path from 'path';
import os from 'os';

describe('pathlib test', () => {
  it('write test', () => {
    const tempDir = fs.mkdtempSync(
      path.join(
        os.tmpdir(),
        'temptemp',
      ),
    );
    try {
      expect(fs.existsSync(tempDir)).toBeTruthy();
    } finally {
      fs.rmdirSync(tempDir, { recursive: true });
      expect(fs.existsSync(tempDir)).toBeFalsy();
    }
  });
});