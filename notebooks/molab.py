import marimo
__generated_with = "0.19.0"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    mo.md("""# Tourism evaluation on MoLab
Attach the GPU, then follow `docs/MOLAB.md`. This notebook does not start long jobs automatically.
Use the terminal to install the repository and execute the resumable CLI. Ordinary generated files
are temporary: export results and feature caches before ending a session.
""")
    return

if __name__ == "__main__":
    app.run()
