# cltl-custom-object-recognition

Two Leolani components that give a conversational agent eyes:

- **`cltl.object_recognition`** — listens for images on the event bus, runs an
  object detector over them, and publishes the detected objects (label,
  bounding box, confidence) plus a scene label and a one-sentence scene
  description as an annotation event.
- **`cltl.situation_awareness`** — keeps a short per-scenario history of the
  scene and the objects that were seen, and what changed (`VisualHistory`).
- **`cltl.visual_responder`** — reads that history and answers questions such as *"what do you see?"* or *"what has
  changed?"* in the chat.

Both attach to a running Leolani deployment through the message bus. Nothing
in the platform needs to know they exist.

## How it fits together

```
   chat UI / camera
        │  cltl.topic.image                    cltl.topic.text_in  (user utterances)
        ▼                                                │
 ┌──────────────────────────┐                            │
 │ ObjectRecognitionService │                            │
 │  fetch image from storage│                            │
 │  ObjectDetector.detect() │                            │
 └────────────┬─────────────┘                            │
              │ cltl.topic.object_recognition            │
              ▼                                          │
 ┌──────────────────────────────┐                        │
 │ SituationAwarenessService    │                        │
 │  VisualHistory.add_event()   │                        │
 └────────────┬─────────────────┘                        │
              │ shared VisualHistory (in process)        │
              ▼                                          ▼
        ┌────────────────────────────────────────────────────┐
        │ VisualResponderService                              │
        │   history = VisualHistory.get(scenario)             │
        │   VisualResponderImpl.respond(utterance, history)   │
        └─────────────────────────┬──────────────────────────┘
                                  │ cltl.topic.vision_out  (agent reply)
                                  ▼
                               chat UI
```

All topic names are configurable, see [Configuration](#configuration).

## Object recognition

`ObjectRecognitionService` (`src/cltl/object_recognition/service.py`)
subscribes to the image topic. An image event carries no pixels, only a
`cltl-storage:image/<id>` reference, so the service fetches the image through
`cltl.backend`'s `ClientImageSource` (configured by `[cltl.backend]
storage_url`). It then calls the configured `ObjectDetector` and publishes an
`ObjectRecognitionEvent` (`schema.py`): one EMISSOR `Mention` per detection,
whose segment is the bounding box in image pixels and whose annotation is an
`Object(type, label, confidence)`. If nothing is detected, a single mention
with a `None` annotation covers the whole image.

### Detector implementations

Selected with `[cltl.object_recognition] implementation`:

| `implementation` | Class | Backend | Scene label | Scene description |
|---|---|---|---|---|
| `llama_cpp` (default) | `LlamaCppObjectDetectorProxy` | A vision-language model (e.g. Qwen3-VL) served by llama.cpp's `llama-server`, via its OpenAI-compatible API | ✓ | ✓ |
| `ollama` | `OllamaObjectDetectorProxy` | A vision-language model (e.g. `qwen2.5vl`) served by a local Ollama, or Ollama's cloud API | ✓ | – |
| `proxy` | `ObjectDetectorProxy` | YOLOv5 in the `tae898/yolov5` Docker image (started automatically when `start_infra: True`) | – | – |

The VLM detectors prompt the model for structured JSON (constrained by a JSON
schema), with boxes normalized to 0–1000, and convert the result to the same
`Object`/`Bounds` shape the YOLO proxy produces, so they are drop-in
replacements. The scene label and description are returned as extra `Object`s
with type `scene` and `scene_description` (`api.SCENE_TYPE`,
`api.SCENE_DESCRIPTION_TYPE`) whose bounds cover the whole image; individual
detections use the model name as type.

`llama_cpp` specifics:

- Images are downscaled to at most 1024 px on the long side before sending, and
  at most 40 objects are requested, so prompt and answer fit in the server's
  context (`-c`). A truncated answer is logged as a warning.
- `frequency_penalty` (default 0.3) keeps the model from repeating the same
  detection over and over.
- `<think>…</think>` output from reasoning models is stripped before parsing.

Start a server, for example:

```bash
llama-server -m Qwen3VL-8B-Instruct-Q4_K_M.gguf \
    --mmproj mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \
    -c 4092 -ngl all --host 0.0.0.0 --port 9009
```

### Trying a detector on a single image

Both VLM detectors have a command line entry point that prints the detections
and writes an annotated copy of the image (`<image>.detections.png`):

```bash
python -m cltl.object_recognition.llama_cpp data/Kyoto2011.JPG --host http://localhost:9009 [--show]
python -m cltl.object_recognition.ollama_proxy data/Kyoto2011.JPG --model qwen2.5vl [--host https://ollama.com --api-key ...]
```

## Visual responder

`SituationAwarenessService` (`src/cltl/situation_awareness/service.py`)
subscribes to the object recognition topic, `VisualResponderService`
(`src/cltl/visual_responder/service.py`) to the text input topic. Both share
one `VisualHistory` (`src/cltl/situation_awareness/history.py`), provided by
`SituationAwarenessContainer`, so other components can use it as well.

**On every object recognition event** the `VisualHistory` summarizes the detections into an
`ImageAnnotation` — scene, scene description and a count per object label —
and appends it to the scenario's history *only if it differs* from the
previous one. The history keeps at most `history_size` annotations; the
oldest drops off.

Because a VLM varies in wording and counts between images of the same view,
annotations are compared leniently (`src/cltl/situation_awareness/api.py`):

- Labels are grouped into categories (`situation_awareness/labels.py`, `CATEGORIES`), so *man*,
  *woman* and *child* all count as *person*, *mug* as *cup*, *couch* as *sofa*,
  and so on. Extend that table as needed; unlisted labels are their own
  category.
- Counts that differ by at most `count_threshold` are considered equal.
- The scene description is never compared, since it is phrased differently
  every time.
- An object that reappears with a count seen earlier in the history is not
  reported as new.

**On every utterance** it calls `VisualResponderImpl.respond()`
(`visualresponder.py`), which answers only if the utterance contains one of
the configured cues as whole words (case-insensitive, apostrophes ignored, so
*what's*, *what’s* and *whats* match alike):

| Cue type | Default cues | Answer |
|---|---|---|
| `see_cues` | look, what do you see, what can you see, … | The scene (*"This looks like a street."*), then either the scene description if nothing changed, or the object counts (*"I see 3 people, a car."*) |
| `change_cues` | what changed, what's new, anything different, differences, … | Scene change, objects that appeared and disappeared (*"Now I also see a dog. I no longer see a bicycle."*) — each change is reported once; asking again gives *"Nothing has changed since I last told you"* |

Change cues are checked first. If no image has been seen yet in the scenario
the reply is *"I don't see anything"*. Any other utterance is ignored, so the
responder can run alongside a general dialogue component on the same input
topic. Replies are published as agent `TextSignalEvent`s on `topic_output`,
with `source=event` so tenant and scenario ids carry over from the question.

## Scenes in the knowledge graph

If `kg_address` is set in `[cltl.situation-awareness]`, `SceneKnowledgeGraph`
(`src/cltl/situation_awareness/knowledge_graph.py`) pushes the scenes seen
during an interaction to the knowledge graph through `cltl.brain`'s
`LongTermMemory.capsule_event`, like the event details in cltl-custom-diabetes.
The capsules are built by `situation_capsule.py`:

- Each scene type is a unique scene in the interaction (scenario), e.g.
  `leolaniWorld:office_<scenario id>`, with a `sem:hasActor` triple for each
  object seen in it.
- The capsule is an experience of the `source` sensor: a `visual` event for the
  scenario with a `detection` per image.
- A scene is pushed the first time it is seen in the interaction. After that it
  is pushed again only if the similarity of its objects to the objects last
  pushed for that scene is below `similarity_threshold`. Similarity is the
  weighted Jaccard of the object counts per category. Images without objects
  are not pushed. This keeps the knowledge graph from filling up with the same
  scene.
- `similarity_threshold` (0 to 1) regulates how many changes are recorded:
  `0.0` pushes each scene type only once per interaction, `1.0` pushes every
  change in the objects of a scene, and values in between push only changes of
  at least that degree. Every image is checked against it, independent of
  `count_threshold`, which only applies to the visual history.
- If `place` is set, the context of each interaction is pushed once with
  `capsule_context`.

## Configuration

`config/default.config` holds the standalone defaults; `config/custom.config`
overrides them for a real deployment. The relevant sections:

```ini
[cltl.event.kombu]
server: $CLTL_AMQP_URL            # broker of the deployment
exchange: cltl.combot
compression: bzip2
tenant: $CLTL_TENANT              # empty = listen to all tenants

[cltl.backend]
storage_url: $CLTL_STORAGE_URL    # where cltl-storage:image/<id> resolves, keep the trailing slash
server_image_url: http://127.0.0.1:8000/host

[cltl.object_recognition]
implementation: llama_cpp         # llama_cpp | ollama | proxy

[cltl.object_recognition.llama_cpp]
model: Qwen3VL-8B-Instruct        # only used as label, llama-server serves one model
host: http://localhost:9009
frequency_penalty: 0.3
# api_key: ...                    # only if llama-server runs with --api-key

[cltl.object_recognition.ollama]
model: qwen2.5vl
# host: https://ollama.com        # default: OLLAMA_HOST or local instance
# api_key: ...                    # default: OLLAMA_API_KEY

[cltl.object_recognition.proxy]
start_infra: True                 # start the YOLOv5 container; False needs detector_url
# detector_url: http://...

[cltl.object_recognition.events]
image_topic: cltl.topic.image
object_topic: cltl.topic.object_recognition

[cltl.situation-awareness]
object_input: cltl.topic.object_recognition
count_threshold: 1                # count differences up to this are ignored
history_size: 10                  # changed annotations kept per scenario (default 5)
kg_address: $CLTL_KG_ADDRESS      # GraphDB repository, scenes are not pushed if unset
kg_log_dir: kg_logs
similarity_threshold: 0.6         # 0 = each scene once, 1 = every change (default 0.6)
source: front-camera
# place: ...                      # with country, region, city: push the interaction context

[cltl.visual-responder]
text_input: cltl.topic.text_in
topic_output: cltl.topic.vision_out
see_cues: look, what do you see, what can you see, what did you see, what have you seen
change_cues: what changed, what has changed, what's changed, what is different, ...
```

`$VAR` values are expanded when read. An unset variable is passed on as the
literal string (with only a warning), so a missing `CLTL_TENANT` makes the
component bind a queue for a tenant literally named `$CLTL_TENANT`.

## Running

Requires Python ≥ 3.10.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Start a detector backend (e.g. `llama-server` as above), make sure the
deployment's broker and storage are reachable, then:

```bash
CLTL_AMQP_URL='amqp://eliza:eliza123@127.0.0.1:5672/' \
CLTL_TENANT=tenant-a \
CLTL_STORAGE_URL=http://127.0.0.1:8001/storage/ \
    python src/main.py
```

`main.py` composes `ObjectRecognitionContainer` and
`VisualResponderContainer`, and serves a `/health` endpoint on port 8006.
Opening and closing scenarios is not its job; that is done by the
deployment's own `cltl-context`.

With `[cltl.event] implementation: internal` (the `default.config` value) the
components use an in-process event bus instead, useful for local testing.

### Docker

```bash
docker build -t cltl-object-recognition --build-context leolani=<path to leolani sdists> .
```

The image installs from an offline registry of first-party sdists (the
`leolani` build context) and ships `config/`, so a deployment only needs to
mount its own `custom.config`.

## Package layout

```
src/
├── main.py                          # application: composes both containers, /health
└── cltl/
    ├── object_recognition/
    │   ├── api.py                   # Object, ObjectDetector, SCENE_TYPE, SCENE_DESCRIPTION_TYPE
    │   ├── container.py             # picks the detector implementation from config
    │   ├── service.py               # image topic → detector → object topic
    │   ├── schema.py                # ObjectRecognitionEvent (EMISSOR mentions)
    │   ├── llama_cpp.py             # VLM via llama-server
    │   ├── ollama_proxy.py          # VLM via Ollama
    │   └── proxy.py                 # YOLOv5 Docker service
    ├── situation_awareness/
    │   ├── api.py                   # ImageAnnotation (change detection)
    │   ├── history.py               # VisualHistory: per-scenario history of changed annotations
    │   ├── knowledge_graph.py       # SceneKnowledgeGraph: pushes changed scenes to the knowledge graph
    │   ├── situation_capsule.py     # capsules for cltl.brain
    │   ├── service.py               # object topic → VisualHistory → knowledge graph
    │   ├── container.py             # visual_history singleton, shared by components
    │   └── labels.py                # label → category table
    └── visual_responder/
        ├── api.py                   # VisualResponder
        ├── container.py
        ├── service.py               # text topic → reply from the VisualHistory
        ├── visualresponder.py       # cue matching and reply generation
        └── framework/               # legacy Pepper robot code, not used
```

## License

MIT — see [`LICENSE`](LICENSE).
