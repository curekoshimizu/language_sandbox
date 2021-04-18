import { using } from 'using-statement';
import path from 'path';
import fs from 'fs';
import readline from 'readline';
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

describe('write data only once then read (string data)', () => {
  it('sync', () => {
    const temp = new TempDir();
    using(temp, () => {
      const fname = path.join(temp.tempDir(), 'file.txt');
      const data = 'party parrot';
      fs.writeFileSync(fname, data);
      const readData = fs.readFileSync(fname, 'utf-8');
      expect(readData).toBe(data);
    });
  });
  it('async', async () => {
    const temp = await AsyncTempDir.new();
    await using(temp, async () => {
      const fname = path.join(temp.tempDir(), 'file.txt');
      const data = 'party parrot';
      await fs.promises.writeFile(fname, data);
      const readData = await fs.promises.readFile(fname, 'utf-8');
      expect(readData).toBe(data);
    });
  });
});

describe('write data 10000 times then read', () => {
  it('async bytes', async () => {
    const temp = await AsyncTempDir.new();
    const count = 10000;

    await using(temp, async () => {
      const fname = path.join(temp.tempDir(), 'file.txt');

      const bufferFromString = (str: string): Buffer => {
        const buffer = Buffer.alloc(str.length);
        buffer.write(str);

        return buffer;
      };

      const origin = 'parrot parrot ';
      const data = bufferFromString(origin);

      await new Promise<void>((resolve) => {
        const stream = fs.createWriteStream(fname);
        stream.on('finish', () => {
          resolve();
        });
        [...Array(count)].forEach(() => {
          stream.write(data);
        });
        stream.end(); // to flush
      });

      const readData = await fs.promises.readFile(fname);
      expect(readData).toEqual(bufferFromString(origin.repeat(count)));
    });
  });
  it('async string', async () => {
    const temp = await AsyncTempDir.new();
    const count = 10000;

    await using(temp, async () => {
      const fname = path.join(temp.tempDir(), 'file.txt');
      const data = 'party parrot';

      await new Promise<void>((resolve) => {
        const stream = fs.createWriteStream(fname);
        stream.on('finish', () => {
          resolve();
        });
        [...Array(count)].forEach(() => {
          stream.write(`${data}\n`);
        });
        stream.end(); // to flush
      });

      const readStream = fs.createReadStream(fname, { encoding: 'utf8' });
      const reader = readline.createInterface({ input: readStream });

      let cnt = 0;
      for await (const line of reader) {
        expect(line).toBe(data);
        cnt += 1;
      }

      expect(cnt).toBe(count);
    });
  });
});
