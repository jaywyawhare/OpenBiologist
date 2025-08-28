# OpenBiologist 🧬

> **PuchAI Hackathon Project** - An MCP-powered WhatsApp bot for accessible biological research and discovery.

## 🌟 About

OpenBiologist is an innovative MCP (Model Context Protocol) application that brings biological research tools directly to WhatsApp. Built for the PuchAI hackathon, this project democratizes access to biological research by allowing users to query databases, analyze sequences, and collaborate on research projects through simple WhatsApp messages.

### Local Development

1. **Clone the repository**
   ```bash
   git clone git@github.com:jaywyawhare/OpenBiologist.git
   cd OpenBiologist
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

4. **Start all services (recommended)**
   ```bash
   # Start both backend and MCP server
   ./run_openbiologist.sh
   ```

## 📊 Roadmap

- [x] **Phase 1**: Core MCP WhatsApp bot
  - [x] Basic project structure
  - [x] WhatsApp Business API integration
  - [x] MCP protocol implementation
  - [x] Basic biological database queries

- [x] **Phase 2**: Advanced biological features
  - [x] Sequence analysis and comparison
  - [x] Protein structure prediction using local AI models (Simplified ESMFold)
  - [ ] Research paper search and summarization

## 🏆 PuchAI Hackathon

This project was developed for the **PuchAI Hackathon**, focusing on:

- **Innovation**: Bringing biological research to WhatsApp through MCP
- **Accessibility**: Making complex biological tools accessible via simple chat
- **AI Integration**: Leveraging MCP protocol for intelligent biological queries
- **Real-world Impact**: Democratizing access to biological research tools

## 📄 License

This project is licensed under the DBaJ-NC-CFL License - see the [LICENSE](./LICENCE.md) file for details.

<div align="center">

**Made with ❤️ for the PuchAI Hackathon**

*Empowering the future of biological research through open collaboration*

</div> 