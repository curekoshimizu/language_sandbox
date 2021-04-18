import util from 'util';

const sleep = (msec: number):Promise<void> => new Promise((resolve) => setTimeout(resolve, msec));
export default sleep;

export const sleep2 = util.promisify(setTimeout);
