const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('scribeSettings', {
  get: () => ipcRenderer.invoke('settings:get'),
  set: (s) => ipcRenderer.invoke('settings:set', s),
});
