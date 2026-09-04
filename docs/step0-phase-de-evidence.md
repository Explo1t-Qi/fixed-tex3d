# Step 0 Phase D/E Runtime Evidence

## Authoritative OpenVLA one-step and rollout smoke

- Branch: `fix/openvla-baseline-correctness`
- Audited source HEAD: `ff9fcbc8f84debb7693027a58490a0d856c8cdde`
- Environment: `tex3d-openvla`
- Checkpoint: `/data/huangsimin/openvla-7b-finetuned-libero-spatial`
- Task/state: `libero_spatial` task 0, training state 0
- Optimization iterations/frames: 1/1
- Rollout episodes: 1
- Run ID: `step0-smoke-EVAL-libero_spatial-2026_09_04-14_10_42`

The one-step metrics were:

| Metric | Observed |
| --- | ---: |
| Action loss | 30.012054443359375 |
| Feature loss | 0.0 |
| Total loss | 30.012054443359375 |
| Image gradient norm max | 9.135553359985352 |
| Texture gradient norm | 0.21750734746456146 |
| Parameter Linf before | 0.0 |
| Parameter Linf after | 0.05000000074505806 |
| Parameter change Linf | 0.05000000074505806 |
| Maximum texture perturbation | 0.025077147409319878 |

The saved noise is a finite `float32 [21932, 3]` tensor with 25,332 nonzero
values and Linf `0.05000000074505806`. The one-element loss history is finite
and equals the reported total loss. A zero feature loss is expected at the
initial zero-texture point; the unchanged action objective supplied the finite
nonzero update gradient.

The frame contract records both texture-sharing instances, visible-pixel counts
of 3,289 and 2,651, a fixed 512 policy-source resolution, a 224 pre-crop canvas,
and deployment crop area 0.9.

The rollout completed one episode, reported success, and logged the saved MP4.
Success is recorded only as a smoke outcome, not as a scientific attack metric.
The synced MP4 is a valid ISO MP4 containing 77 decodable 512 x 512 RGB frames
at 30 FPS (2.57 seconds).

Key artifact SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| `Ep0_step_metrics.jsonl` | `82e1ad667309d531a4a71980b6c245e6ec2f84c906ffd7dfa4260d56f0b1fe74` |
| `Ep0_frame_contract.json` | `318cdb523952ec6a2eb5dd69ba7be6118c8d393ed249a10f6fbcb98e2248149e` |
| `Ep0_gradient_log.txt` | `296f6f99228ad17639cf30ee8f71c7f69a8f59955a2dbb8588f1940cd225c3e3` |
| `Ep0_Vertex_Noise.pt` | `7099a4639a803b3f4b853aae05b3ee01840788ca57bd361fe1dce5f5baf238cd` |
| `Ep0_UV_Map.png` | `e70238ccfd0cf2acb9ee26b24f9638a005ce0fe7f48a886f3caeeffedcafe7dc` |
| `Ep0_loss_history.npy` | `4f163e7b72cc4b06f96c216c72e3eb5556ae2f5922ea76b10a82aef25398c3b6` |
| Rollout MP4 | `0132d72946515a12fe0a00a044c8532d40b8bc8a6bd85dc0ec99982d07c86e57` |

## Joint-environment compatibility smoke

A second complete run used the OpenPI-oriented joint environment under run ID
`step0-smoke-EVAL-libero_spatial-2026_09_04-14_17_02`. Its log explicitly
warned that `transformers 4.53.2` and `tokenizers 0.21.4` differ from OpenVLA's
required `4.40.1` and `0.19.1`. It is therefore supplementary compatibility
evidence and does not replace the authoritative OpenVLA processor audit.

It nevertheless completed one update and one rollout with finite metrics:

```text
action/total loss       28.125
feature loss            0
image gradient max      12.152493476867676
texture gradient norm   0.1645379662513733
parameter change Linf   0.05000000074505806
max texture perturbation 0.025077147409319878
rollout                 complete, reported success
```

The different action loss is retained as evidence that dependency variants are
not treated as numerically interchangeable.

## Cleanup and repository state

The post-run audit was performed after the joint compatibility run:

```text
HEAD ff9fcbc8f84debb7693027a58490a0d856c8cdde
git status --short: empty
remaining clean-backup search: empty
```

Its SHA-256 is
`00ee478f321171b723f074b5b34533b896e35d84fca26fe4ca7079f81ebc8ece`.
Both complete runs logged final restoration of the original XML and real
MuJoCo texture. The cleanup audit confirms that the final server asset state is
clean and no temporary backup remains.

## Verdict

```text
PHASE D PASS
PHASE E PASS
```

These are baseline correctness smoke results. They do not establish attack
effectiveness and do not authorize Step 1.
