# nodejs sandbox

## How did I initialize this project?

```
npm init
npm i -D typescript @types/node ts-node ts-node-dev
npx tsc --init
```

then, edit tsconfig.json

## How to execute

`npm run dev` or `npx ts-node src/index.ts`

watch mode (hot reload)

`npm run dev:watch` or `npx npx ts-node-dev (--poll) --respawn -- src/index.ts`


## How to introduce eslint

```
npm i -D eslint
npx eslint --init
```

setting file: `.eslintrc.json`

## How to introduce jest

```
npm i -D jest @types/jest ts-jest -D
```

setting file: `jest.config.js`
