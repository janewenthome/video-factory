import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

const source = readFileSync(new URL('../src/root.tsx', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    esModuleInterop: true,
    jsx: ts.JsxEmit.ReactJSX,
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2020,
  },
}).outputText;

const module = {exports: {}};
const mocks = {
  'react': {},
  'react/jsx-runtime': {jsx: () => null, jsxs: () => null, Fragment: Symbol('Fragment')},
  'remotion': {},
  '@remotion/media': {},
};
const load = new Function('exports', 'require', 'module', compiled);
load(module.exports, (name) => {
  if (!(name in mocks)) throw new Error(`Unexpected import in renderer test: ${name}`);
  return mocks[name];
}, module);

const audioVolume = module.exports.audioVolume;
const closeTo = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} != ${expected}`);

test('music overlaps natural sound and ducks smoothly, then returns to its base level', () => {
  const music = {type: 'music', timeline_start: 0, timeline_end: 60};
  const ranges = [[21, 24.5], [27.5, 31.5], [43, 47]];
  const at = (seconds) => audioVolume(music, 30, seconds * 30, ranges);

  closeTo(at(15), 0.2);
  closeTo(at(20.75), 0.2);
  closeTo(at(20.875), 0.124);
  closeTo(at(21), 0.048);
  closeTo(at(22), 0.048);
  closeTo(at(24.5), 0.048);
  closeTo(at(24.625), 0.124);
  closeTo(at(24.75), 0.2);
  closeTo(at(25), 0.2);
  closeTo(at(43), 0.048);
  closeTo(at(47.75), 0.2);
});

test('natural source audio stays full level while explicitly muted video stays silent', () => {
  const naturalVideo = {type: 'video', timeline_start: 21, timeline_end: 24.5, audio: {preserve_natural: true}};
  const mutedVideo = {type: 'video', timeline_start: 0, timeline_end: 5, audio: {preserve_natural: false}};

  closeTo(audioVolume(naturalVideo, 30, 15), 1);
  closeTo(audioVolume(mutedVideo, 30, 15), 0);
});
