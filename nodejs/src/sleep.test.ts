import sleep, { sleep2 } from './sleep';

describe('sleep test', () => {
  it('sleep / date', async () => {
    const startTime = new Date().getTime();
    await sleep(100);
    const elapsedTime = new Date().getTime() - startTime;
    expect(elapsedTime).toBeGreaterThanOrEqual(99);
    expect(elapsedTime).toBeLessThan(120);
  });

  it('sleep 2 / date', async () => {
    const startTime = new Date().getTime();
    await sleep2(100);
    const elapsedTime = new Date().getTime() - startTime;
    expect(elapsedTime).toBeGreaterThanOrEqual(99);
    expect(elapsedTime).toBeLessThan(120);
  });

  it('sleep / nodejs elapsed time', async () => {
    const startTime = process.hrtime();
    await sleep(50);
    const elapsedTime = process.hrtime(startTime); // [sec, nano]
    expect(elapsedTime[0]).toBe(0);
    expect(elapsedTime[1]).toBeGreaterThan(49 * 1000000);
    expect(elapsedTime[1]).toBeLessThan(52 * 1000000);
  });

  it('sleep 2 / nodejs elapsed time', async () => {
    const startTime = process.hrtime();
    await sleep2(50);
    const elapsedTime = process.hrtime(startTime); // [sec, nano]
    expect(elapsedTime[0]).toBe(0);
    expect(elapsedTime[1]).toBeGreaterThan(49 * 1000000);
    expect(elapsedTime[1]).toBeLessThan(52 * 1000000);
  });
});

describe('async test', () => {
  const lazyAdd = async (x: number, y: number, msec: number = 10):Promise<number> => {
    await sleep(msec);

    return x + y;
  };

  it('async gather', async () => {
    const [x, y] = await Promise.all([lazyAdd(1, 1, 10), lazyAdd(1, 2)]);
    const ret = await lazyAdd(x, y);
    expect(ret).toBe(5);
  });

  /* eslint func-style: off */
  async function* genNumbers(): AsyncGenerator<number> {
    await sleep(10);
    yield 1;
    await sleep(10);
    yield 2;
    await sleep(10);
    yield 3;
  }

  it('async generator', async () => {
    let sum = 0;
    for await (const x of genNumbers()) {
      sum += x;
    }
    expect(sum).toBe(6);
  });
});
