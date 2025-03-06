"""
Debug tool to help analyze and fix LLM response parsing issues.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.llm_utils import extract_json_from_llm_response

def analyze_llm_response(response_text: str) -> dict:
    """
    Analyze an LLM response to identify issues preventing JSON extraction.
    
    Args:
        response_text: The raw text response from an LLM
        
    Returns:
        Dictionary containing analysis results
    """
    analysis = {
        "original_length": len(response_text),
        "first_50_chars": response_text[:50] + "..." if len(response_text) > 50 else response_text,
        "has_triple_backticks": "```" in response_text,
        "issues": []
    }
    
    # Check for common issues
    if not response_text.strip():
        analysis["issues"].append("Empty response")
    
    # Check for incomplete JSON (unmatched braces)
    open_braces = response_text.count("{")
    close_braces = response_text.count("}")
    if open_braces != close_braces:
        analysis["issues"].append(f"Unmatched braces: {open_braces} opening vs {close_braces} closing")
    
    # Check for triple backtick formatting
    backtick_matches = re.findall(r"```(?:json)?([\s\S]*?)```", response_text)
    if backtick_matches:
        analysis["backtick_blocks"] = len(backtick_matches)
        
        # Check content inside backticks
        for i, content in enumerate(backtick_matches):
            try:
                json.loads(content.strip())
                analysis["valid_json_in_block"] = i + 1
            except json.JSONDecodeError as e:
                analysis["issues"].append(f"JSON decode error in block {i+1}: {str(e)}")
    else:
        analysis["backtick_blocks"] = 0
    
    # Try to extract JSON using our utility
    extracted_json = extract_json_from_llm_response(response_text)
    analysis["json_extracted"] = extracted_json is not None
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Debug LLM response JSON extraction issues")
    parser.add_argument("filename", help="File containing LLM response text")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues")
    args = parser.parse_args()
    
    if not os.path.exists(args.filename):
        print(f"Error: File not found - {args.filename}")
        return
    
    with open(args.filename, 'r', encoding='utf-8') as f:
        response_text = f.read()
    
    # Analyze the response
    analysis = analyze_llm_response(response_text)
    
    # Print analysis results
    print("=== LLM Response Analysis ===")
    print(f"Length: {analysis['original_length']} characters")
    print(f"Preview: {analysis['first_50_chars']}")
    print(f"Triple backtick blocks: {analysis['backtick_blocks']}")
    print(f"JSON extraction successful: {analysis['json_extracted']}")
    
    if analysis["issues"]:
        print("\nIssues found:")
        for issue in analysis["issues"]:
            print(f"- {issue}")
    
    # Attempt to fix if requested
    if args.fix and not analysis["json_extracted"]:
        print("\nAttempting to fix issues...")
        
        # Try various fixes
        fixed_text = response_text
        
        # Fix 1: Add missing braces if needed
        open_braces = fixed_text.count("{")
        close_braces = fixed_text.count("}")
        if open_braces > close_braces:
            fixed_text += "}" * (open_braces - close_braces)
            print(f"Added {open_braces - close_braces} closing braces")
        elif close_braces > open_braces:
            fixed_text = "{" * (close_braces - open_braces) + fixed_text
            print(f"Added {close_braces - open_braces} opening braces")
        
        # Fix 2: Wrap in backticks if needed
        if "```" not in fixed_text:
            fixed_text = "```json\n" + fixed_text + "\n```"
            print("Wrapped content in triple backticks")
        
        # Try to extract JSON from fixed text
        fixed_json = extract_json_from_llm_response(fixed_text)
        if fixed_json is not None:
            print("\nFixed JSON successfully extracted!")
            output_filename = args.filename + ".fixed"
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(fixed_text)
            print(f"Fixed content saved to {output_filename}")
        else:
            print("\nCould not fix JSON extraction issues automatically.")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
