"""typed_ir variant for TaskBench *resource* domains (HuggingFace, Multimedia).

Failure analysis (27B dep4+retrieval, HF seed 1, 57/100 fail): 50/57 were TOOL CHOICE -
extra helper/preprocessing steps (image-to-text, object detection, VQA) and near-miss
tasks (document QA vs VQA, segmentation vs detection); only 7 were edge errors. Gold uses
the single most specific task per requested operation, and links only real data flow.
So: no forced decomposition, no default mention-order chaining, precise task choice.
"""
from __future__ import annotations

from tpj.ir.formats.typed_ir import TypedIRFormat

INSTRUCTIONS_DEP5R = """\
Write the plan as JSON: {"steps": [...], "final": [...]}.
Each step has:
- "id": "s1", "s2", ... (unique)
- "tool": exact tool name from the tool list
- "args": argument values. Use user-given files/text as literals (e.g. "example.jpg"). To feed the
  OUTPUT of an earlier step into this one write {"$ref": "sK"}.
- "returns": the type the step produces (e.g. "image", "text", "audio", "video")
- "safety": "safe" for these model/media tools unless they change the world
- "after": ids of the step(s) whose output this step consumes.

[1. ONE TOOL PER REQUESTED OPERATION] Use exactly one step for each operation the user asks for,
choosing the MOST SPECIFIC tool for that operation. Do NOT add helper or preprocessing steps the user
did not ask for (e.g. do not run object detection, image-to-text or captioning first just to "help"
answer a question) - if one tool directly produces what is asked, use only that tool. Do not repeat
the same tool for the same input.

[2. CHOOSE THE PRECISE TASK] Distinguish closely related tasks by what the user wants:
  - question about the content of a document, form, receipt, table or text in an image ->
    Document Question Answering; question about a natural photo/scene -> Visual Question Answering;
    question answered from a given text passage -> Question Answering.
  - separate/outline/mask regions or objects -> Image Segmentation; find/locate/count objects with
    boxes -> Object Detection; assign one label to the whole image -> Image Classification.
  - describe an image in words -> Image-to-Text; edit an image according to an instruction ->
    Image Editing; transform an image's style or appearance -> Image-to-Image.
  - transcribe speech to text -> Automatic Speech Recognition; label parts of a sentence/entities ->
    Token Classification; open-ended dialogue -> Conversational.
  Use the exact tool names from the tool list.

[3. DEPENDENCIES = DATA FLOW ONLY] Link step B to step A only when B processes A's OUTPUT (e.g.
"translate the transcription", "then make a video of the edited image"): pass {"$ref":"sA"} in B's
argument and list "sA" in B's "after". If B works on the user's original file or text, give it that
literal input and leave "after" empty, even if B is mentioned after A. Do not chain independent
operations.

Example (hypothetical request) - "Transcribe 'example.wav', translate the transcription to French,
and separately tell me what objects are in 'example.jpg'":
{"steps":[
 {"id":"s1","tool":"Automatic Speech Recognition","args":{"audio":"example.wav"},"returns":"text","safety":"safe","after":[]},
 {"id":"s2","tool":"Translation","args":{"text":{"$ref":"s1"},"target":"French"},"returns":"text","safety":"safe","after":["s1"]},
 {"id":"s3","tool":"Object Detection","args":{"image":"example.jpg"},"returns":"objects","safety":"safe","after":[]}
],"final":["s2","s3"]}

"final": ids of the steps whose outputs answer the request.
If no tool is needed, return {"steps": [], "final": []}."""


class TypedIRDep5RFormat(TypedIRFormat):
    name = "typed_ir_dep5r"

    def instructions(self) -> str:
        return INSTRUCTIONS_DEP5R
