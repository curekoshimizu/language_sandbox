import fs from 'fs';
import path from 'path';
import os from 'os';

export default class TempDir {
    private tmpDir: string;

    constructor() {
      this.tmpDir = fs.mkdtempSync(
        path.join(
          os.tmpdir(),
          'tempdir',
        ),
      );
    }

    tempDir(): string {
      return this.tmpDir;
    }

    dispose():void {
      fs.rmdirSync(this.tmpDir, { recursive: true });
    }
}

export class AsyncTempDir {
    private tmpDir: string;

    constructor(tmpDir: string) {
      this.tmpDir = tmpDir;
    }

    static async new():Promise<AsyncTempDir> {
      const tmpDir = await fs.promises.mkdtemp(
        path.join(
          os.tmpdir(),
          'tempdir',
        ),
      );

      return new AsyncTempDir(tmpDir);
    }

    tempDir(): string {
      return this.tmpDir;
    }

    async dispose():Promise<void> {
      await fs.promises.rmdir(this.tmpDir, { recursive: true });
    }
}
