# Regulatory Data Sources

## Corpus Scope

| Regulation | Official reference | Corpus role | Status note |
| --- | --- | --- | --- |
| PP Nomor 5 Tahun 2021 | [BPK Regulation Database](https://peraturan.bpk.go.id/Details/161835/pp) | Risk-based business licensing | No longer in force; repealed by PP Nomor 28 Tahun 2025. Retained only for the historical baseline. |
| PP Nomor 35 Tahun 2021 | [BPK Regulation Database](https://peraturan.bpk.go.id/Details/161904/pp-no-35-tahun-2021) | Fixed-term employment, outsourcing, working time, and termination | Verify current status before each corpus release. |
| PP Nomor 51 Tahun 2023 | [BPK Regulation Database](https://peraturan.bpk.go.id/Details/270269/pp-no-51-tahun-2023) | Wage regulation changes | Amended by PP Nomor 49 Tahun 2025; retained only for the historical baseline. |
| UU Nomor 6 Tahun 2023 | [BPK Regulation Database](https://peraturan.bpk.go.id/Details/246523/uu-no-6-tahun-2023) | Job creation legal framework | The official record lists Constitutional Court decisions that affect how several provisions must be read. |

The original notebook downloads working copies from a Google Drive folder. Those copies are historical experiment inputs, not the authoritative distribution channel.

The four-document M1 benchmark therefore measures the historical project corpus. Questions asking for current law must return `insufficient_context` until a separately versioned, effective corpus is acquired and reviewed.

## Acquisition Policy

Production ingestion must obtain documents from an official government source and record:

- Canonical regulation identifier and title.
- Source URL.
- Retrieval timestamp.
- SHA-256 checksum.
- Effective-status check timestamp.
- Amendment or judicial-review relationships.

PDF files are not committed to Git. A corpus release must be reproducible from its source manifest and checksums.

## Usage Basis

Article 42(b) of Indonesian Law Number 28 of 2014 on Copyright states that statutory regulations are not protected by copyright. Website layouts, compiled metadata, and non-regulatory supporting material may have separate terms; preserve attribution and comply with the source website's access terms.

This repository does not redistribute the source PDFs. Any deployment must verify that its document acquisition and distribution process remains compliant with applicable terms and current law.
