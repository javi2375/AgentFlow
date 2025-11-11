import os
import json
import requests
import ssl
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()

from agentflow.agentflow.tools.base import BaseTool

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "AI_Model_Search_Tool"

LIMITATIONS = """
1. This tool is specifically designed for searching information about AI models.
2. It may not be as effective for general web searches outside of AI models.
3. Information accuracy depends on the reliability of sources found.
4. Some AI model information may be behind paywalls or require special access.
"""

BEST_PRACTICES = """
1. Use this tool when you need information about specific AI models, their capabilities, or release dates.
2. Be specific in your queries (e.g., "Zhipu AI latest model" instead of just "AI models").
3. For the most recent information, include time-related terms like "latest", "2024", or "recent".
4. Cross-reference information from multiple sources when possible.
"""

class AI_Model_Search_Tool(BaseTool):
    def __init__(self, model_string=None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A specialized search tool for finding information about AI models, with focus on model specifications, release dates, capabilities, and comparisons.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query specifically about AI models.",
                "search_sources": "list - Optional list of sources to search. Default includes multiple AI news sites and company pages.",
            },
            output_type="str - Comprehensive information about the requested AI model.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Zhipu AI latest model 2024")',
                    "description": "Search for information about Zhipu AI's latest model released in 2024."
                },
                {
                    "command": 'execution = tool.execute(query="OpenAI GPT-5 release date capabilities")',
                    "description": "Search for information about OpenAI's GPT-5 release date and capabilities."
                },
                {
                    "command": 'execution = tool.execute(query="Anthropic Claude 3.5 vs GPT-4 comparison")',
                    "description": "Search for comparison between Claude 3.5 and GPT-4 models."
                }
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            }
        )
        self.max_retries = 3
        self.timeout = 15  # seconds
        
        # Default sources for AI model information
        self.default_sources = [
            "https://www.zhipuai.cn",
            "https://github.com",
            "https://huggingface.co",
            "https://arxiv.org",
            "https://www.theverge.com",
            "https://techcrunch.com",
            "https://www.wired.com",
            "https://www.technologyreview.com",
            "https://www.anthropic.com",
            "https://openai.com",
            "https://blog.google",
            "https://ai.meta.com"
        ]

    def _search_source(self, source_url, query):
        """
        Search a specific source for information about AI models.
        This is a simplified implementation - in a real scenario, you might use site-specific search APIs.
        """
        try:
            # For demonstration, we'll use a general web search approach
            # In practice, you might implement site-specific search logic
            
            # Create headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Use a search endpoint if available, otherwise just return source URL
            if "zhipuai" in source_url:
                # Special handling for Zhipu AI with SSL verification bypass
                search_url = f"https://www.zhipuai.cn/api/v4/model/list"  # Example API endpoint
                try:
                    # Create an SSL context that doesn't verify certificates for zhipuai.cn
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    
                    response = requests.get(
                        search_url, 
                        headers=headers, 
                        timeout=self.timeout,
                        verify=False  # Disable SSL verification for zhipuai.cn due to certificate issues
                    )
                    if response.status_code == 200:
                        return self._parse_zhipuai_response(response.json(), query)
                except Exception as e:
                    # If API fails, provide helpful information about Zhipu AI
                    return self._get_zhipuai_fallback_info(query)
            
            # Fallback: return information about where to search
            return f"For information about '{query}', check {source_url} directly or use a general search engine."
            
        except Exception as e:
            return f"Error searching {source_url}: {str(e)}"

    def _parse_zhipuai_response(self, response_data, query):
        """
        Parse Zhipu AI API response for model information.
        """
        try:
            if "data" in response_data and isinstance(response_data["data"], list):
                models = response_data["data"]
                latest_model = None
                latest_date = None
                
                for model in models:
                    if "created_at" in model and model["created_at"]:
                        model_date = model["created_at"]
                        if latest_date is None or model_date > latest_date:
                            latest_date = model_date
                            latest_model = model
                
                if latest_model:
                    return f"Latest Zhipu AI model: {latest_model.get('name', 'Unknown')} (Released: {latest_date}). Details: {latest_model.get('description', 'No description available')}"
            
            return "Could not find specific model information from Zhipu AI API."
        except Exception as e:
            return f"Error parsing Zhipu AI response: {str(e)}"
    
    def _get_zhipuai_fallback_info(self, query):
        """
        Provide fallback information about Zhipu AI when direct API access fails.
        """
        return f"""
For information about Zhipu AI's latest models, here are the best approaches:

1. **Official Sources**:
   - Zhipu AI official website: https://www.zhipuai.cn
   - Zhipu AI GitHub: https://github.com/zhipuai
   - Zhipu AI documentation: https://open.bigmodel.cn/dev/api

2. **Latest Known Models**:
   - GLM-4: Zhipu's flagship language model with strong reasoning capabilities
   - GLM-4V: Multimodal version with vision capabilities
   - GLM-3-Turbo: Faster, more efficient version for common tasks
   - CodeGeeX: Code generation model

3. **Alternative Access Methods**:
   - Use a web search with terms like "Zhipu AI GLM-4 release date 2024"
   - Check AI news sites that cover Chinese AI companies
   - Look for announcements on Zhipu's official WeChat account
   - Check Hugging Face: https://huggingface.co/models?search=zhipuai

4. **For API Access**:
   - Try using the Chinese domain (zhipuai.cn) instead of international
   - Consider using a VPN if accessing from outside China
   - Contact Zhipu AI directly for API access issues

Query: {query}
        """

    def _search_general_web(self, query):
        """
        Fallback to general web search when specific sources don't yield results.
        """
        try:
            # Use a general search API or service
            # For this example, we'll return a formatted response about where to find information
            search_terms = query.lower()
            
            if "zhipuai" in search_terms and ("latest" in search_terms or "new" in search_terms):
                return """
For the latest information about Zhipu AI models, I recommend checking:
1. Official Zhipu AI website: https://www.zhipuai.cn
2. Zhipu AI's GitHub repository: https://github.com/zhipuai
3. Recent AI news sites that cover Chinese AI companies
4. Hugging Face model hub: https://huggingface.co/models?search=zhipuai

Zhipu AI (智谱AI) is known for their GLM (General Language Model) series, including GLM-4, GLM-4V, and other models. They frequently release updates, so checking their official channels is recommended for the most current information.
                """
            
            return f"Search for '{query}' in AI-specific sources. For comprehensive results, try searching multiple AI-focused websites and official company pages."
            
        except Exception as e:
            return f"Error in general web search: {str(e)}"

    def execute(self, query: str, search_sources: list = None):
        """
        Execute AI model search.
        
        Parameters:
            query (str): The search query specifically about AI models.
            search_sources (list): Optional list of sources to search.
        
        Returns:
            str: Comprehensive information about the requested AI model.
        """
        sources = search_sources if search_sources else self.default_sources
        
        # Special handling for Zhipu AI queries
        if "zhipuai" in query.lower() and ("latest" in query.lower() or "new" in query.lower()):
            # Try to get information from Zhipu AI directly
            zhipuai_result = self._search_source("https://www.zhipuai.cn", query)
            if not zhipuai_result.startswith("Error") and not zhipuai_result.startswith("For information"):
                return zhipuai_result
        
        # Try other sources
        for source in sources[:3]:  # Limit to first 3 sources for efficiency
            result = self._search_source(source, query)
            if result and not result.startswith("Error") and not result.startswith("For information"):
                return result
        
        # Fallback to general web search
        return self._search_general_web(query)

    def get_metadata(self):
        """
        Returns metadata for AI_Model_Search_Tool.
        
        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    """
    Test:
    cd agentflow/tools/ai_model_search
    python tool.py
    """
    def print_json(result):
        import json
        print(json.dumps(result, indent=4))
    
    ai_model_search = AI_Model_Search_Tool()
    
    # Get tool metadata
    metadata = ai_model_search.get_metadata()
    print("Tool Metadata:")
    print_json(metadata)
    
    examples = [
        {'query': 'Zhipu AI latest model 2024'},
        {'query': 'OpenAI GPT-5 release date'},
        {'query': 'Anthropic Claude 3.5 specifications'},
    ]
    
    for example in examples:
        print(f"\nExecuting search: {example['query']}")
        try:
            result = ai_model_search.execute(**example)
            print("Search Result:")
            print(result)
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)
    
    print("Done!")