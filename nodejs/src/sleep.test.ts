import sleep from './sleep';

test('sleep / date', async () => {
  const startTime = new Date().getTime();
  await sleep(100);
  const elapsedTime = new Date().getTime() - startTime;
  expect(elapsedTime).toBeGreaterThanOrEqual(100);
  expect(elapsedTime).toBeLessThan(120);
});

test('sleep / nodejs elapsed time', async () => {
  const startTime = process.hrtime();
  await sleep(10);
  const elapsedTime = process.hrtime(startTime); // [sec, nano]
  expect(elapsedTime[0]).toBe(0);
  expect(elapsedTime[1]).toBeGreaterThan(10 * 1000000);
  expect(elapsedTime[1]).toBeLessThan(12 * 1000000);
});

const lazyAdd = async (x: number, y: number):Promise<number> => {
  await sleep(10);

  return x + y;
};

test('async gather', async () => {
  const [x, y] = await Promise.all([lazyAdd(1, 1), lazyAdd(1, 2)]);
  const ret = await lazyAdd(x, y);
  expect(ret).toBe(5);
});
