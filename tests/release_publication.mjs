// Exercise the installed release plugin with an owned GitHub API boundary.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const { publish } = await import(pathToFileURL(resolve(process.argv[2], 'index.js')));
const config = JSON.parse(readFileSync('.releaserc.json', 'utf8'));
const [, options] = config.plugins.find(([name]) => name === '@semantic-release/github');
const releases = [];

class OwnedGitHub {
  async request(route, payload) {
    assert.equal(route, 'POST /repos/{owner}/{repo}/releases');
    assert.equal(payload.owner, 'fixture');
    assert.equal(payload.repo, 'release-check');
    const release = { ...payload, draft: payload.draft ?? false, id: 1,
      html_url: 'https://example.invalid/owned-release' };
    releases.push(release);
    return { data: release };
  }
}

await publish(options, {
  cwd: process.cwd(),
  env: { GH_TOKEN: 'synthetic-unused', GITHUB_ACTION: 'owned-check' },
  options: { repositoryUrl: 'https://github.com/fixture/release-check' },
  branch: { name: 'main', type: 'release' },
  nextRelease: { gitTag: 'v0.0.0-probe', version: '0.0.0', name: 'Owned release', notes: 'Owned notes' },
  logger: { log() {}, error() {} },
}, { Octokit: OwnedGitHub });

assert.equal(releases.length, 1, 'The release plugin must prepare a release');
assert.equal(releases[0].draft, true, 'Stable releases must stay private until installers are attached');
assert.equal(releases[0].body, 'Owned notes');
assert.equal(releases[0].tag_name, 'v0.0.0-probe');
console.log('✅ Stable release stays private while installers are built');
