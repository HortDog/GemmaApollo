// GemmaApollo Scribe desktop shell (electron-3tier plan, stage 5).
//
// The shell is deliberately thin: spawn the Python app server (the sidecar)
// on a free loopback port, wait until it answers, and load its URL — the
// renderer IS frontend/index.html served by the sidecar, one origin, so ws
// and getUserMedia need no extra plumbing (127.0.0.1 is a secure context).
//
// Dev mode (not packaged, or SCRIBE_DEV=1): spawns `uv run scribe serve`
// from the repo checkout. Packaged: runs the PyInstaller-frozen
// resources/scribe-server/scribe-server.exe (stage 6).
const { app, BrowserWindow, Menu, dialog, ipcMain, session } = require('electron');
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const http = require('http');
const net = require('net');
const path = require('path');

const DEFAULTS = { modelServerUrl: 'http://127.0.0.1:8018' };
const isPackaged = app.isPackaged && !process.env.SCRIBE_DEV;
let child = null;
let childErr = '';
let port = null;
let win = null;
let settingsWin = null;

// ---------- settings (JSON in userData, passed to the sidecar as flags) ----
const settingsPath = () => path.join(app.getPath('userData'), 'settings.json');
function loadSettings() {
  try { return { ...DEFAULTS, ...JSON.parse(fs.readFileSync(settingsPath(), 'utf8')) }; }
  catch { return { ...DEFAULTS }; }
}
function saveSettings(s) {
  fs.mkdirSync(path.dirname(settingsPath()), { recursive: true });
  fs.writeFileSync(settingsPath(), JSON.stringify(s, null, 2));
}

// ---------- sidecar lifecycle ----------
function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const p = srv.address().port;
      srv.close(() => resolve(p));
    });
    srv.on('error', reject);
  });
}

function sidecarCmd(p, settings) {
  const args = ['serve', '--engine', 'remote', '--port', String(p),
                '--model-server-url', settings.modelServerUrl];
  if (isPackaged) {
    return { cmd: path.join(process.resourcesPath, 'scribe-server', 'scribe-server.exe'),
             args, cwd: path.join(process.resourcesPath, 'scribe-server') };
  }
  return { cmd: 'uv', args: ['run', 'scribe', ...args],
           cwd: path.join(__dirname, '..') };            // repo root
}

function waitReady(url, timeoutMs) {
  const t0 = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const retry = () => (Date.now() - t0 > timeoutMs)
        ? reject(new Error(`scribe server did not answer within ${timeoutMs / 1000}s`))
        : setTimeout(tick, 250);
      http.get(url, res => { res.resume(); res.statusCode < 500 ? resolve() : retry(); })
          .on('error', retry);
    };
    tick();
  });
}

async function startSidecar(settings) {
  port = await freePort();
  const { cmd, args, cwd } = sidecarCmd(port, settings);
  childErr = '';
  child = spawn(cmd, args, {
    cwd,
    stdio: ['pipe', 'pipe', 'pipe'],   // stdin pipe = the sidecar's orphan guard
    env: { ...process.env, SCRIBE_PARENT_WATCH: '1' },
    shell: process.platform === 'win32' && !isPackaged,  // resolve `uv` via PATH
  });
  child.stderr.on('data', d => { childErr = (childErr + d).slice(-4000); });
  child.stdout.on('data', d => process.stdout.write(d));
  child.on('exit', code => {
    if (code && !app.isQuitting) {
      dialog.showErrorBox('scribe server exited',
                          childErr || `exit code ${code}`);
    }
  });
  try {
    await waitReady(`http://127.0.0.1:${port}/`, 30000);
  } catch (e) {
    throw new Error(`${e.message}\n\n${childErr}`);
  }
}

function killSidecar() {
  if (!child) return;
  try { child.stdin.end(); } catch {}                   // EOF → sidecar self-exits
  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F']);
  } else {
    child.kill();
  }
  child = null;
}

async function restartSidecar() {
  killSidecar();
  await startSidecar(loadSettings());
  if (win) win.loadURL(`http://127.0.0.1:${port}/`);
}

// ---------- settings window ----------
function openSettings() {
  if (settingsWin) { settingsWin.focus(); return; }
  settingsWin = new BrowserWindow({
    width: 460, height: 220, parent: win, modal: true, resizable: false,
    autoHideMenuBar: true, title: 'Scribe settings',
    webPreferences: { contextIsolation: true,
                      preload: path.join(__dirname, 'preload.js') },
  });
  settingsWin.loadFile(path.join(__dirname, 'settings.html'));
  settingsWin.on('closed', () => { settingsWin = null; });
}

ipcMain.handle('settings:get', () => loadSettings());
ipcMain.handle('settings:set', async (_ev, s) => {
  saveSettings({ ...loadSettings(), ...s });
  await restartSidecar();                               // new flags need a new sidecar
  return loadSettings();
});

function buildMenu() {
  return Menu.buildFromTemplate([
    {
      label: 'Scribe',
      submenu: [
        { label: 'Model server…', click: openSettings },
        { label: 'Restart scribe server',
          click: () => restartSidecar().catch(e => dialog.showErrorBox('restart failed', String(e))) },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    { label: 'View', submenu: [{ role: 'reload' }, { role: 'toggleDevTools' },
                               { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'resetZoom' }] },
  ]);
}

// ---------- app lifecycle ----------
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => { if (win) { win.restore(); win.focus(); } });

  app.whenReady().then(async () => {
    // mic permission for our own UI; everything else denied
    session.defaultSession.setPermissionRequestHandler((_wc, permission, cb) => {
      cb(permission === 'media');
    });
    Menu.setApplicationMenu(buildMenu());
    try {
      await startSidecar(loadSettings());
    } catch (e) {
      dialog.showErrorBox('GemmaApollo Scribe failed to start', String(e.message || e));
      app.quit();
      return;
    }
    win = new BrowserWindow({ width: 1280, height: 860,
                              webPreferences: { contextIsolation: true } });
    win.on('closed', () => { win = null; });
    if (process.env.SCRIBE_SMOKE) {                     // scripted self-test
      win.webContents.on('did-finish-load', () => {
        console.log(`SMOKE OK port=${port}`);
        app.quit();
      });
      win.webContents.on('did-fail-load', (_e, code, desc) => {
        console.error(`SMOKE FAIL ${code} ${desc}`);
        app.exit(1);
      });
    }
    win.loadURL(`http://127.0.0.1:${port}/`);
  });

  app.on('will-quit', () => { app.isQuitting = true; killSidecar(); });
  app.on('window-all-closed', () => app.quit());
}
