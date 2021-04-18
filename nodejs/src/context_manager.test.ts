import { using } from 'using-statement';

class Disposable {
    done = false;

    dispose():void {
      this.done = true;
    }
}

class AsyncDisposable {
    done = false;

    dispose():Promise<void> {
      return new Promise<void>((resolve) => {
        setTimeout(() => {
          this.done = true;
          resolve();
        }, 10);
      });
    }
}

describe('context manager', () => {
  it('sync', () => {
    const disposable = new Disposable();
    using(disposable, () => {
      expect(disposable.done).toBeFalsy();
    });
    expect(disposable.done).toBeTruthy();
  });
  it('async', async () => {
    const disposable = new AsyncDisposable();
    await using(disposable, () => {
      expect(disposable.done).toBeFalsy();
    });
    expect(disposable.done).toBeTruthy();
  });
});
