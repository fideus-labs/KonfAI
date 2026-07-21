// SPDX-License-Identifier: Apache-2.0

// Decode a Server-Sent-Events response into parsed `data:` frames. Both the chat turn and the live job
// stream consume this and keep their own per-event dispatch.
export async function* readSSE(resp: Response): AsyncGenerator<any> {
  const reader = resp.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (!chunk.startsWith("data: ")) continue;
      yield JSON.parse(chunk.slice(6));
    }
  }
}
