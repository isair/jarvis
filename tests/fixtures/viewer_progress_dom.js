const escapeText = text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
class Element {
    constructor() {
        this._html = '';
        this.children = [];
        this.textContent = '';
        this.style = {};
        this.dataset = {};
        this.handlers = {};
        this.scrollHeight = 100;
    }
    set innerHTML(value) { this._html = value; this.children = []; }
    get innerHTML() { return this._html + this.children.map(child => `<div>${escapeText(child.textContent)}</div>`).join(''); }
    appendChild(child) { this.children.push(child); }
    addEventListener(event, handler) { this.handlers[event] = handler; }
    remove() {}
}
const elements = new Map();
const document = {
    body: new Element(),
    createElement: () => new Element(),
    querySelector: () => null,
    getElementById: id => {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
    }
};
let diaryImportDone = false;
const loadGraphData = () => {}, loadTreeData = () => {}, loadStats = () => {}, showToast = () => {};
const fetch = async () => {
    let sent = false;
    return {body: {getReader: () => ({read: async () => sent ? {done: true} :
        (sent = true, {done: false, value: new TextEncoder().encode(streamText)})})}};
};
