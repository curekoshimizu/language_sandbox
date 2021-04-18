import { using } from 'using-statement';
import fs from 'fs';
import TempDir, { AsyncTempDir } from './fs_temp';

describe('temp', () => {
  it('temp dir', () => {
    const temp = new TempDir();
    using(temp, () => {
      expect(fs.existsSync(temp.tempDir())).toBeTruthy();
    });
    expect(fs.existsSync(temp.tempDir())).toBeFalsy();
  });
  it('async temp dir', async () => {
    const temp = await AsyncTempDir.new();
    await using(temp, () => {
      expect(fs.existsSync(temp.tempDir())).toBeTruthy();
    });
    expect(fs.existsSync(temp.tempDir())).toBeFalsy();
  });
});
