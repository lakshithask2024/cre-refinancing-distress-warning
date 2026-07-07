# Power BI Reports

Place the built `.pbix` file here after constructing it in Power BI Desktop.

## File

- `cre_distress_dashboard.pbix` — Main portfolio risk dashboard (5 pages)

## How to Build

1. Follow `../05_reviewer_quickstart.md` to load data
2. Follow `../01_data_model.md` to set up relationships
3. Follow `../02_dax_measures.md` to add all measures
4. Follow `../03_report_pages.md` to build each page

## Git LFS

The `.pbix` file is binary and can be large (10-50 MB). Consider using Git LFS:

```bash
git lfs track "*.pbix"
git add .gitattributes
```
