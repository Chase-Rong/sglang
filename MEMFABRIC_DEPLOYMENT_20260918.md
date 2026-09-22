# MemFabric A5 deployment on the three test hosts

Installed in `jzx_sglang_5.3_flash` on 141.61.55.34, 141.61.54.248 and
141.61.54.244. No compilation was required. The validated A5 binaries were
repacked as `memfabric_hybrid-1.2.1+a5.3ef081a7-cp312-cp312-linux_aarch64.whl`.
The local version suffix records the optional Host URMA symbol compatibility
repair. This is a locally packaged build, not a new upstream release.

## Installed locations

- Python package: `/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid`.
- Both A5 extension libraries: `memfabric_hybrid/extend` under that package.
- Setup script: `/usr/local/memfabric_hybrid/set_env.sh`.
- Matching FSA binding: `/usr/local/memfabric_hybrid/1.2.1-a5-3ef081a7/fsa/libsgl_fsa_extension.so`.
- Deployment files and reports: `/home/j00628475/memfabric_deployment_20260917`.

Python defaults to the installed MemFabric package. The installed `.pth` file
sets the extension-library path only if the process has no explicit override.
For service launch, source the setup script after CANN. It configures the
MemFabric library paths and provides the new FSA binding path only when
`SGLANG_NPU_FSA_EXTENSION` is not already set. Existing launchers that explicitly
select another Python package or FSA binding must be reviewed separately.
SGLang service source paths and attention OPP vendors were not changed here.

## Validation

All three installations verified31 installed-file hashes, imported the new
package/version, loaded both extension libraries and the standalone FSA binding,
and checked `/proc/self/maps` for loaded MemFabric libraries. All resolved to
the installed package, with no candidate `/home` or old wheel-library paths.

| Host | Import/native load | NPU initialization and DMA roundtrip |
|---|---|---|
| 141.61.55.34 | Passed | Not rerun: all eight cards occupied by an existing service |
| 141.61.54.248 | Passed | Passed on idle NPU0, TaskQueue1,1GiB LOCAL pool |
| 141.61.54.244 | Passed | Passed on idle NPU0, TaskQueue1,1GiB LOCAL pool |

248 and244 were started from stopped containers; only their existing
`sleep infinity` container command and verification processes were started.
No inference service was started or stopped.34's existing service was preserved;
installation does not establish what libraries an already-running process uses.
No verification processes remain. This validates installed LOCAL DMA support;
it is not a new PD transport or full-model performance qualification.

The compatibility build still warns that optional Host URMA symbols are absent.
248/244 also warn about optional `libhcom.so`; LOCAL initialization and DMA
completed successfully. Host URMA capabilities are not qualified by these tests.

## Backups

Each directory contains `previous_installation.tar.gz` and `paths.json`:

| Host | Backup directory under the deployment root |
|---|---|
|34|`backup_20260918_094849`|
|248|`backup_20260918_095215`|
|244|`backup_20260918_095227`|

Keep these archives for rollback after draining users. Do not extract old files
on top of an active new package; remove the new package through its installation
metadata first and restore the recorded original paths during a maintenance window.

## Provenance

Wheel SHA256: `6b294c268e840ebaf8c36c016d519566b2be2b753e29af4db11d7aebe8211ceb`.
Core SHA256: `73a34fff8693ea6f0791df088572b8efc0b65c070e1bd21b0f17a886a38bf5e8`.
Accoffload extension: `605949d0c4abcae08893d423949378b6ab8bc8a882cf9b32f1b0fa5dc336b8a6`.
Copy extension: `edc8e506a3050b2a0f567d5a313ad14568dc76ff8ab5ee558572c8a7d72109df`.
FSA binding: `0db9ef1fbe67d3e8dd1a571b9d7fa18c02efd7e7786eed496a8f757c8589ffa7`.

Local artifacts and per-host JSON/logs:
`results/unified_host_20260917/deployment_package/`.
