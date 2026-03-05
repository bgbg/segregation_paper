"""Combine multiple markdown files into a single academic paper.

This script combines the individual markdown files from the reports directory
into a single cohesive paper with proper formatting, cross-references, and
image path handling. It can generate both markdown and PDF/Word outputs.

Usage:
    python combine_paper.py [--output-format FORMAT] [--output-dir DIR] [--title TITLE] [--verbose]

Output formats:
    - markdown: Single combined .md file
    - pdf: PDF using pandoc with xelatex
    - word: Microsoft Word .docx file
    - all: Generate all formats

The script hardcodes the file order and handles:
- Cross-references between sections
- Image path adjustments
- Table of contents generation
- Proper academic formatting
"""

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import defopt


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_markdown_file(file_path: Path) -> str:
    """Load a markdown file and return its content."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Markdown file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")


def adjust_image_paths(content: str, base_dir: Path, output_dir: Path) -> str:
    """Adjust image paths to be relative to the output markdown file location."""
    # Pattern to match markdown image syntax: ![alt](path)
    image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

    def replace_image_path(match):
        alt_text = match.group(1)
        image_path = match.group(2)

        # If it's a relative path from plots/, resolve it relative to base_dir
        # and then make it relative to output_dir
        if image_path.startswith("plots/"):
            # The plots are in transition_paper/plots/, resolve to output location
            full_plot_path = base_dir / image_path
            try:
                # Make relative to the output directory where pandoc will run
                rel_path = os.path.relpath(full_plot_path, output_dir)
                return f"![{alt_text}]({rel_path})"
            except ValueError:
                # If we can't make it relative, keep original
                return match.group(0)

        # If it's an absolute path, convert to relative from the output location
        if os.path.isabs(image_path):
            try:
                # Make relative to the output_dir (where pandoc will run)
                rel_path = os.path.relpath(image_path, output_dir)
                return f"![{alt_text}]({rel_path})"
            except ValueError:
                # If we can't make it relative, keep original
                return match.group(0)

        # For other relative paths, ensure they're relative to the output location
        return f"![{alt_text}]({image_path})"

    return re.sub(image_pattern, replace_image_path, content)


def format_figure_captions(content: str) -> str:
    """Convert figure captions to proper format for Word documents.

    Converts:
    ![alt](path) *Figure X: description*

    To:
    ![alt](path){#fig:label}

    Figure X: description
    """
    # Pattern to match image followed by italic figure caption
    # This handles multi-line captions that start with *Figure and end with *
    fig_pattern = r"(!\[[^\]]*\]\([^)]+\))\s*\*((?:Figure\s+\d+:[^*]*(?:\n[^*]*)*?))\*"

    def replace_figure(match):
        image_markdown = match.group(1)
        caption_text = match.group(2)

        # Extract figure number for label
        fig_match = re.search(r"Figure\s+(\d+)", caption_text)
        if fig_match:
            fig_num = fig_match.group(1)
            fig_label = fig_num
        else:
            fig_label = "figure"

        # Clean up caption text (remove the * markers)
        clean_caption = caption_text.strip()

        # Format as proper figure with caption
        return f"{image_markdown}{{#fig:{fig_label}}}\n\n{clean_caption}\n"

    return re.sub(fig_pattern, replace_figure, content, flags=re.MULTILINE | re.DOTALL)


def generate_table_of_contents(content: str) -> str:
    """Generate a table of contents from markdown headers."""
    lines = content.split("\n")
    toc_lines = ["# Table of Contents", ""]

    for line in lines:
        # Match headers (##, ###, etc.)
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2)

            # Create anchor link
            anchor = re.sub(r"[^\w\s-]", "", title.lower())
            anchor = re.sub(r"[-\s]+", "-", anchor)

            # Indent based on header level
            indent = "  " * (level - 1)
            toc_lines.append(f"{indent}- [{title}](#{anchor})")

    toc_lines.append("")
    return "\n".join(toc_lines)


def combine_markdown_files(
    input_files: List[Path], output_path: Path, base_dir: Path, title: str
) -> str:
    """Combine multiple markdown files into a single document."""
    logger = logging.getLogger(__name__)

    combined_content = []
    title_added = False
    toc_added = False

    for file_path in input_files:
        logger.info(f"Processing {file_path.name}")

        content = load_markdown_file(file_path)

        # Add title and table of contents after the first file (Introduction)
        if not title_added and file_path.name.startswith("01_"):
            # Add the title at the beginning
            combined_content.append(f"# {title}")
            combined_content.append("")

            # Generate and add table of contents
            toc = generate_table_of_contents(content)
            combined_content.append(toc)
            title_added = True
            toc_added = True

        # Adjust image paths to be relative to output directory
        output_dir = output_path.parent
        content = adjust_image_paths(content, base_dir, output_dir)

        # Format figure captions for proper Word formatting
        content = format_figure_captions(content)

        # Add content with section separator
        combined_content.append(content)
        combined_content.append("")  # Add blank line between sections

    # Join all content
    final_content = "\n".join(combined_content)

    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_content, encoding="utf-8")

    logger.info(f"Combined markdown written to: {output_path}")
    return final_content


def convert_to_pdf(markdown_path: Path, output_path: Path) -> None:
    """Convert markdown to PDF using pandoc."""
    logger = logging.getLogger(__name__)

    # Check if pandoc is available
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("pandoc is not installed or not available in PATH")

    # Check if xelatex is available
    try:
        subprocess.run(["xelatex", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("xelatex not found, falling back to pdflatex")
        pdf_engine = "pdflatex"
    else:
        pdf_engine = "xelatex"

    # Build pandoc command
    # Use relative paths since we'll run pandoc from the markdown directory
    markdown_rel = markdown_path.name
    output_rel = output_path.relative_to(markdown_path.parent)

    cmd = [
        "pandoc",
        markdown_rel,
        "-o",
        str(output_rel),
        f"--pdf-engine={pdf_engine}",
        "-V",
        "geometry:margin=1in",
        "-V",
        "fontsize=11pt",
        "-V",
        "documentclass=article",
        "--toc",  # Generate table of contents
        "--toc-depth=3",  # Include up to level 3 headers
        "--number-sections",  # Number sections
        "--citeproc",  # Process citations
    ]

    # Add crossref filter if available
    try:
        subprocess.run(
            ["pandoc-crossref", "--version"], capture_output=True, check=True
        )
        cmd.extend(["--filter", "pandoc-crossref"])
        logger.info("Using pandoc-crossref for cross-references")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("pandoc-crossref not available, skipping cross-references")

    try:
        logger.info(f"Converting to PDF using {pdf_engine}...")
        # Run pandoc from the directory containing the markdown file
        # This ensures that relative image paths are resolved correctly
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=markdown_path.parent
        )
        logger.info(f"PDF generated: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"PDF conversion failed: {e}")
        logger.error(f"Pandoc stderr: {e.stderr}")
        raise RuntimeError(f"PDF conversion failed: {e.stderr}")


def convert_to_word(markdown_path: Path, output_path: Path) -> None:
    """Convert markdown to Word document using pandoc."""
    logger = logging.getLogger(__name__)

    # Check if pandoc is available
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("pandoc is not installed or not available in PATH")

    # Build pandoc command for Word
    # Use relative paths since we'll run pandoc from the markdown directory
    markdown_rel = markdown_path.name
    output_rel = output_path.relative_to(markdown_path.parent)

    cmd = [
        "pandoc",
        markdown_rel,
        "-o",
        str(output_rel),
        "--toc",  # Generate table of contents
        "--toc-depth=3",  # Include up to level 3 headers
        "--number-sections",  # Number sections
        "--citeproc",  # Process citations
    ]

    try:
        logger.info("Converting to Word document...")
        # Run pandoc from the directory containing the markdown file
        # This ensures that relative image paths are resolved correctly
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=markdown_path.parent
        )
        logger.info(f"Word document generated: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Word conversion failed: {e}")
        logger.error(f"Pandoc stderr: {e.stderr}")
        raise RuntimeError(f"Word conversion failed: {e.stderr}")


def main(
    *,
    output_format: Literal["markdown", "pdf", "word", "all"] = "all",
    output_dir: str = "data/processed/reports",
    title: str = "From Separation to Transition: Tracking Electoral Flows in Israel's Ultra-Orthodox Sector",
    verbose: bool = False,
) -> int:
    """Combine markdown files into a single academic paper.

    Args:
        output_format: Output format ('markdown', 'pdf', 'word', or 'all')
        output_dir: Directory for output files
        title: Title of the paper
        verbose: Enable verbose logging

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger = setup_logging(verbose)

    # Define input files in order
    reports_dir = Path("transition_paper")
    input_files = [
        reports_dir / "00_abstract.md",
        reports_dir / "01_intro.md",
        reports_dir / "02_methods.md",
        reports_dir / "03_results.md",
        reports_dir / "04_conclusions.md",
        reports_dir / "05_endmatter.md",
        reports_dir / "10_references.md",
        reports_dir / "09_appendix.md",
    ]

    # Verify all input files exist
    missing_files = [f for f in input_files if not f.exists()]
    if missing_files:
        logger.error(f"Missing input files: {[f.name for f in missing_files]}")
        return 1

    # Set up output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate base filename
    base_filename = "haredi_voter_transitions_paper"

    try:
        # Always generate markdown first
        markdown_path = output_dir_path / f"{base_filename}.md"
        logger.info("Combining markdown files...")

        combine_markdown_files(input_files, markdown_path, reports_dir, title)

        # Generate additional formats based on request
        if output_format in ["pdf", "all"]:
            pdf_path = output_dir_path / f"{base_filename}.pdf"
            convert_to_pdf(markdown_path, pdf_path)

        if output_format in ["word", "all"]:
            word_path = output_dir_path / f"{base_filename}.docx"
            convert_to_word(markdown_path, word_path)

        logger.info("Paper generation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Paper generation failed: {e}")
        return 1


if __name__ == "__main__":
    defopt.run(
        main,
        short={
            "output-format": "f",
            "output-dir": "d",
            "title": "t",
            "verbose": "v",
        },
    )
