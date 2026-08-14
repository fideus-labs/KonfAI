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
      let event: any;
      try {
        event = JSON.parse(chunk.slice(6));
      } catch {
        // One frame the parser cannot read must not take the rest of the stream with it: throwing here
        // ends the generator, the consumer reconnects, replays the same frame and throws again, so the
        // feed dies on the spot and never recovers. Drop the frame, keep reading.
        continue;
      }
      yield event;
    }
  }
}
