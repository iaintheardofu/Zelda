# ZELDA - Claude Code Specialized Agents

## 🎯 Overview

ZELDA now has access to **95 specialized AI agents** for comprehensive development support across all aspects of the project.

---

## 📊 Agent Categories

### 🔧 Development Team (10 agents)
- **frontend-developer** - React, Next.js, TypeScript frontend development
- **backend-architect** - Backend system design and architecture
- **fullstack-developer** - Full-stack application development
- **mobile-developer** - iOS and Android development
- **ios-developer** - iOS-specific development
- **ui-ux-designer** - User interface and experience design
- **cli-ui-designer** - Command-line interface design
- **devops-engineer** - DevOps practices and automation
- **frontend-developer** - Frontend specialization

### 🛠️ Development Tools (11 agents)
- **code-reviewer** - Code quality and best practices review
- **debugger** - Bug identification and fixing
- **context-manager** - Project context management
- **error-detective** - Error detection and resolution
- **test-engineer** - Testing strategies and implementation
- **mcp-expert** - Model Context Protocol expertise
- **command-expert** - CLI command optimization
- **performance-profiler** - Performance analysis
- **dx-optimizer** - Developer experience optimization

### 💾 Database (7 agents)
- **database-architect** - Database schema design
- **database-admin** - Database administration
- **database-optimizer** - Query and performance optimization
- **database-optimization** - Optimization strategies
- **supabase-schema-architect** - Supabase-specific schema design
- **sql-pro** - SQL expert
- **graphql-architect** - GraphQL API design

### 🐍 Programming Languages (8 agents)
- **python-pro** - Python expertise
- **typescript-pro** - TypeScript best practices
- **javascript-pro** - JavaScript mastery
- **golang-pro** - Go language expert
- **php-pro** - PHP development
- **shell-scripting-pro** - Bash/shell scripting
- **rust-pro** - Rust programming
- **sql-pro** - SQL expertise

### 🤖 AI Specialists (7 agents)
- **prompt-engineer** - AI prompt optimization
- **task-decomposition-expert** - Breaking down complex tasks
- **search-specialist** - Search and retrieval optimization
- **model-evaluator** - AI model evaluation
- **llms-maintainer** - Large language model maintenance
- **ai-engineer** - AI engineering
- **ml-engineer** - Machine learning engineering

### 📊 Data & AI (7 agents)
- **data-scientist** - Data analysis and modeling
- **data-engineer** - Data pipeline engineering
- **data-analyst** - Data analysis
- **quant-analyst** - Quantitative analysis
- **computer-vision-engineer** - Computer vision systems
- **mlops-engineer** - ML operations

### 🏗️ DevOps & Infrastructure (9 agents)
- **deployment-engineer** - Deployment strategies
- **cloud-architect** - Cloud infrastructure design
- **devops-troubleshooter** - DevOps issue resolution
- **security-engineer** - Security implementation
- **network-engineer** - Network configuration
- **monitoring-specialist** - System monitoring
- **terraform-specialist** - Infrastructure as code

### 🔒 Security (3 agents)
- **security-auditor** - Security audits
- **api-security-audit** - API security testing
- **penetration-tester** - Penetration testing

### 📝 Documentation (6 agents)
- **api-documenter** - API documentation
- **technical-writer** - Technical documentation
- **documentation-expert** - Documentation strategies
- **changelog-generator** - Automated changelog generation
- **docusaurus-expert** - Docusaurus documentation sites

### ⚡ Performance Testing (4 agents)
- **performance-engineer** - Performance optimization
- **react-performance-optimization** - React-specific optimization
- **react-performance-optimizer** - React performance
- **test-automator** - Test automation

### 🌐 Web Tools (4 agents)
- **nextjs-architecture-expert** - Next.js architecture
- **seo-analyzer** - SEO analysis and optimization
- **web-accessibility-checker** - WCAG compliance
- **react-performance-optimizer** - React optimization

### 🔬 Deep Research Team (11 agents)
- **technical-researcher** - Technical research
- **research-orchestrator** - Research coordination
- **research-coordinator** - Multi-agent research
- **academic-researcher** - Academic research
- **fact-checker** - Fact verification
- **research-synthesizer** - Research synthesis
- **report-generator** - Report generation
- **agent-overview** - Agent capability overview
- **research-brief-generator** - Research brief creation
- **competitive-intelligence-analyst** - Competitive analysis
- **data-analyst** - Data-driven insights

### 💼 Business & Marketing (6 agents)
- **product-strategist** - Product strategy
- **business-analyst** - Business analysis
- **content-marketer** - Content marketing
- **payment-integration** - Payment system integration
- **legal-advisor** - Legal compliance

### 🎯 Expert Advisors (4 agents)
- **architect-review** - Architecture review
- **documentation-expert** - Documentation best practices
- **agent-expert** - AI agent expertise
- **dependency-manager** - Dependency management

### 🔧 MCP Development Team (3 agents)
- **mcp-server-architect** - MCP server design
- **mcp-integration-engineer** - MCP integration
- **mcp-deployment-orchestrator** - MCP deployment

### 📄 OCR Extraction Team (2 agents)
- **markdown-syntax-formatter** - Markdown formatting
- **document-structure-analyzer** - Document analysis

### 📓 Obsidian Ops Team (2 agents)
- **review-agent** - Review workflows
- **connection-agent** - Connection management

### 🎙️ Podcast Creator Team (1 agent)
- **project-supervisor-orchestrator** - Project supervision

### 🔄 Modernization (1 agent)
- **architecture-modernizer** - Legacy system modernization

### 🔗 Git (1 agent)
- **git-flow-manager** - Git workflow management

### ⚡ Realtime (1 agent)
- **supabase-realtime-optimizer** - Supabase Realtime optimization

---

## 🚀 Usage

### Access Any Agent

Agents are stored in `.claude/agents/` as markdown files. Each agent provides specialized expertise for specific tasks.

**Example Commands:**
```bash
# List all agents
ls .claude/agents/

# View a specific agent
cat .claude/agents/typescript-pro.md

# Search for agents by topic
grep -l "database" .claude/agents/*.md
```

### Agent Invocation

Agents can be invoked through Claude Code's task system. The system automatically selects the most appropriate agent(s) for each task.

---

## 📋 Key Capabilities by Project Area

### ZELDA Frontend (Next.js)
**Relevant Agents:**
- nextjs-architecture-expert
- typescript-pro
- frontend-developer
- react-performance-optimizer
- ui-ux-designer
- web-accessibility-checker
- seo-analyzer

### ZELDA Backend (Python + Supabase)
**Relevant Agents:**
- python-pro
- backend-architect
- database-architect
- supabase-schema-architect
- supabase-realtime-optimizer
- api-documenter
- security-auditor

### RF Signal Processing & ML
**Relevant Agents:**
- ai-engineer
- ml-engineer
- data-scientist
- computer-vision-engineer
- mlops-engineer
- performance-profiler
- python-pro

### TDOA Geolocation
**Relevant Agents:**
- python-pro
- ai-engineer
- data-engineer
- performance-engineer
- test-engineer

### Countermeasure System
**Relevant Agents:**
- backend-architect
- python-pro
- security-engineer
- api-security-audit
- test-automator

### Database & Schema
**Relevant Agents:**
- supabase-schema-architect
- database-architect
- database-optimizer
- sql-pro
- graphql-architect

### DevOps & Deployment
**Relevant Agents:**
- deployment-engineer
- cloud-architect
- devops-engineer
- terraform-specialist
- monitoring-specialist
- network-engineer

### Documentation
**Relevant Agents:**
- technical-writer
- api-documenter
- documentation-expert
- changelog-generator
- docusaurus-expert

### Testing & QA
**Relevant Agents:**
- test-engineer
- test-automator
- performance-engineer
- security-auditor
- penetration-tester
- api-security-audit

### Code Review & Quality
**Relevant Agents:**
- code-reviewer
- architect-review
- typescript-pro
- python-pro
- javascript-pro
- error-detective
- debugger

---

## 🎯 Specialized ZELDA Use Cases

### 1. Waterfall Display Optimization
**Use:** react-performance-optimizer, performance-profiler, frontend-developer
**Task:** Optimize Canvas rendering for 60 FPS waterfall

### 2. TDOA Algorithm Enhancement
**Use:** ai-engineer, python-pro, data-scientist
**Task:** Improve phase-shift TDOA accuracy

### 3. Threat Classification ML Model
**Use:** ml-engineer, ai-engineer, model-evaluator
**Task:** Enhance threat detection accuracy to 97%+

### 4. Supabase Schema Optimization
**Use:** supabase-schema-architect, database-optimizer, sql-pro
**Task:** Optimize database queries and indexes

### 5. Countermeasure System Testing
**Use:** test-engineer, test-automator, security-auditor
**Task:** Comprehensive countermeasure testing

### 6. API Security Audit
**Use:** api-security-audit, security-engineer, penetration-tester
**Task:** Security review of edge functions

### 7. Performance Profiling
**Use:** performance-profiler, performance-engineer, react-performance-optimization
**Task:** Identify and fix performance bottlenecks

### 8. Documentation Generation
**Use:** technical-writer, api-documenter, documentation-expert
**Task:** Generate comprehensive API docs

### 9. Deployment Automation
**Use:** deployment-engineer, devops-engineer, terraform-specialist
**Task:** Automated deployment pipeline

### 10. Code Review
**Use:** code-reviewer, typescript-pro, python-pro, architect-review
**Task:** Comprehensive code quality review

---

## 📊 Agent Statistics

**Total Agents:** 95

**By Category:**
- Development Team: 10
- Development Tools: 11
- Database: 7
- Programming Languages: 8
- AI Specialists: 7
- Data & AI: 7
- DevOps & Infrastructure: 9
- Security: 3
- Documentation: 6
- Performance Testing: 4
- Web Tools: 4
- Deep Research: 11
- Business & Marketing: 6
- Expert Advisors: 4
- Other Categories: 8

---

## 🔧 Integration with ZELDA

All agents are integrated into the ZELDA development workflow and can be invoked for:
- Code reviews
- Architecture decisions
- Performance optimization
- Security audits
- Documentation
- Testing
- Deployment
- Research
- Debugging

**The agents work collaboratively to provide comprehensive development support across the entire ZELDA platform.**

---

## 📞 Support

**Agent Directory:** `.claude/agents/`
**Documentation:** This file + individual agent markdown files

---

**ZELDA v2.0 - Powered by 95 Specialized AI Agents**

🤖 Generated with [Claude Code](https://claude.com/claude-code)
