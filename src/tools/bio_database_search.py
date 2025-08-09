from typing import Annotated
from pydantic import Field
import httpx
from src.utils.models import RichToolDescription

BioDatabaseSearchDescription = RichToolDescription(
    description="Search biological databases for protein sequences, structures, and related information",
    use_when="Use this when user wants to search for protein information in biological databases",
    side_effects="Queries external biological databases and returns formatted results",
)

async def search_protein_database(
    query: Annotated[str, Field(description="Search query (protein name, gene name, accession number, etc.)")],
    database: Annotated[str, Field(description="Database to search (uniprot, pdb, ncbi, or all)")] = "all",
    max_results: Annotated[int, Field(description="Maximum number of results to return")] = 5,
) -> str:
    """
    Search biological databases for protein information.
    
    This tool searches multiple biological databases to find protein sequences,
    structures, and related information. Supported databases include:
    - UniProt: Protein sequences and annotations
    - PDB: 3D protein structures
    - NCBI: Gene and protein information
    
    Args:
        query: Search term (protein name, gene name, accession, etc.)
        database: Specific database to search or "all" for comprehensive search
        max_results: Maximum number of results to return
    
    Returns:
        Formatted search results from biological databases
    """
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = []
        
        # UniProt search
        if database in ["uniprot", "all"]:
            try:
                uniprot_url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=json&size={max_results}"
                response = await client.get(uniprot_url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("results"):
                        results.append("## 🧬 **UniProt Results**")
                        for entry in data["results"][:max_results]:
                            results.append(f"""
**Entry**: {entry.get('primaryAccession', 'N/A')}
**Name**: {entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')}
**Gene**: {entry.get('genes', [{}])[0].get('geneName', {}).get('value', 'N/A')}
**Organism**: {entry.get('organism', {}).get('scientificName', 'N/A')}
**Length**: {entry.get('sequence', {}).get('length', 'N/A')} amino acids
**Sequence**: {entry.get('sequence', {}).get('value', 'N/A')[:50]}...
""")
            except Exception as e:
                results.append(f"❌ **UniProt Error**: {str(e)}")
        
        # PDB search
        if database in ["pdb", "all"]:
            try:
                pdb_url = f"https://data.rcsb.org/rest/v1/core/search?q={query}&wt=json&rows={max_results}"
                response = await client.get(pdb_url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("result"):
                        results.append("## 🏗️ **PDB Structure Results**")
                        for entry in data["result"]["docs"][:max_results]:
                            results.append(f"""
**PDB ID**: {entry.get('pdb_id', 'N/A')}
**Title**: {entry.get('title', 'N/A')}
**Method**: {entry.get('experimental_method', 'N/A')}
**Resolution**: {entry.get('resolution', 'N/A')} Å
**Organism**: {entry.get('organism_scientific_name', 'N/A')}
**Chains**: {entry.get('number_of_chains', 'N/A')}
""")
            except Exception as e:
                results.append(f"❌ **PDB Error**: {str(e)}")
        
        # NCBI search
        if database in ["ncbi", "all"]:
            try:
                ncbi_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=protein&term={query}&retmode=json&retmax={max_results}"
                response = await client.get(ncbi_url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("esearchresult", {}).get("idlist"):
                        results.append("## 🔬 **NCBI Protein Results**")
                        for protein_id in data["esearchresult"]["idlist"][:max_results]:
                            # Get protein details
                            detail_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={protein_id}&rettype=fasta&retmode=text"
                            detail_response = await client.get(detail_url)
                            if detail_response.status_code == 200:
                                fasta_content = detail_response.text
                                if fasta_content:
                                    lines = fasta_content.split('\n')
                                    if len(lines) >= 2:
                                        header = lines[0]
                                        sequence = ''.join(lines[1:])
                                        results.append(f"""
**NCBI ID**: {protein_id}
**Header**: {header}
**Sequence Length**: {len(sequence)} amino acids
**Sequence Preview**: {sequence[:50]}...
""")
            except Exception as e:
                results.append(f"❌ **NCBI Error**: {str(e)}")
        
        if not results:
            return f"""🔍 **No Results Found**

**Query**: "{query}"
**Database**: {database}

No matching results were found in the biological databases.
Try:
- Using a different search term
- Checking spelling
- Using accession numbers (e.g., P12345, 1ABC)
- Using gene names (e.g., TP53, BRCA1)"""
        
        return f"""🔍 **Biological Database Search Results**

**Query**: "{query}"
**Database**: {database}
**Results**: {len(results) - 1} entries found

{''.join(results)}

💡 **Tips**:
- Use accession numbers for precise results
- Gene names often return multiple entries
- Check multiple databases for comprehensive information
- Use the sequence data for structure prediction""" 