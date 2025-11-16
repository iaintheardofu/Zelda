# ZELDA - Automation Tools & MCP Servers Guide

## 🎯 Overview

ZELDA now has **63 productivity and automation components** installed:
- **27 Automation Hooks** - Code quality, git workflow, security, testing
- **34 MCP Servers** - Database, browser automation, DevOps integrations
- **2 Skills** - Theme factory, enterprise communications

---

## 🪝 Installed Hooks (27)

### Automation (6 hooks)
- **simple-notifications** - Desktop notifications for build/deploy events
- **dependency-checker** - Alerts for outdated dependencies
- **agents-md-loader** - Auto-loads agent context from `.claude/agents/`
- **build-on-change** - Triggers builds when files change
- **deployment-health-monitor** - Monitors deployment status
- **vercel-environment-sync** - Syncs env vars with Vercel

### Git Workflow (5 hooks)
- **smart-commit** - AI-generated commit messages
- **auto-git-add** - Automatically stages related files
- **validate-branch-name** - Enforces branch naming conventions
- **prevent-direct-push** - Blocks direct pushes to main/master
- **conventional-commits** - Enforces conventional commit format

### Development Tools (6 hooks)
- **lint-on-save** - Runs linter when files are saved
- **change-tracker** - Tracks file changes across sessions
- **smart-formatting** - Auto-formats code on save
- **command-logger** - Logs all Claude Code commands
- **file-backup** - Creates backups before destructive operations
- **nextjs-code-quality-enforcer** - Next.js-specific quality rules

### Post-tool Hooks (4 hooks)
- **format-python-files** - Auto-formats Python after edits
- **format-javascript-files** - Auto-formats JS/TS after edits
- **git-add-changes** - Stages changes after tool runs
- **run-tests-after-changes** - Runs tests after code changes

### Pre-tool Hooks (2 hooks)
- **update-search-year** - Updates search year context
- **backup-before-edit** - Backs up files before editing

### Security (2 hooks)
- **security-scanner** - Scans for security vulnerabilities
- **file-protection** - Prevents editing sensitive files

### Testing (1 hook)
- **test-runner** - Automatically runs test suites

### Performance (1 hook)
- **performance-monitor** - Tracks performance metrics

---

## 🔌 MCP Servers (34)

### Integration (2 servers)
**GitHub Integration**
```json
"github": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_TOKEN>"
  }
}
```
- Repository operations
- Issue/PR management
- Code search across repos

**Memory Integration**
```json
"memory": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-memory"]
}
```
- Persistent context storage
- Cross-session memory

### Database (3 servers)
**Supabase**
```json
"supabase": {
  "command": "npx",
  "args": [
    "-y",
    "@supabase/mcp-server-supabase@latest",
    "--read-only",
    "--project-ref=<project-ref>"
  ],
  "env": {
    "SUPABASE_ACCESS_TOKEN": "<personal-access-token>"
  }
}
```
- Database queries
- Schema inspection
- Realtime subscriptions

**PostgreSQL**
```json
"postgresql": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-postgres"],
  "env": {
    "POSTGRES_CONNECTION_STRING": "postgresql://user:password@localhost:5432/dbname"
  }
}
```

**MySQL**
```json
"mysql": {
  "command": "uvx",
  "args": ["mcp-server-mysql"],
  "env": {
    "MYSQL_CONNECTION_STRING": "mysql://user:password@localhost:3306/dbname"
  }
}
```

### Browser Automation (6 servers)
- **playwright-server** - Playwright automation
- **playwright-mcp-server** (executeautomation) - Enhanced Playwright
- **automatalabs-playwright-server** - Automata Labs variant
- **browsermcp** - Browser automation via @browsermcp
- **browserbase** - Cloud browser automation
- **browser-server** - Browser-use MCP server

### DeepGraph Code Analysis (4 servers)
- **DeepGraph React** - React codebase analysis
- **DeepGraph Next.js** - Next.js codebase analysis
- **DeepGraph TypeScript** - TypeScript codebase analysis
- **DeepGraph Vue** - Vue.js codebase analysis

### DevTools (13 servers)
- **chrome-devtools** - Chrome DevTools integration
- **context7** - Upstash Context7 integration
- **ios-simulator-mcp** - iOS Simulator control
- **markitdown** - Markdown conversion
- **serena** - Development assistant
- **Figma Dev Mode** - Figma API integration
- **codacy** - Code quality analysis
- **TestSprite** - Automated testing
- **dynatrace** - Application monitoring
- **azure-kubernetes-service** - AKS management
- **box** - Box.com integration
- **launchdarkly** - Feature flags
- **pulumi** - Infrastructure as Code
- **jfrog** - Artifact management
- **logfire** - Logging and monitoring

### Productivity (1 server)
- **monday-api-mcp** - Monday.com project management

### Audio (1 server)
- **elevenlabs** - Text-to-speech generation

### Web (1 server)
- **fetch** - HTTP request capabilities

### Filesystem (1 server)
- **filesystem** - File system access

---

## 💡 Skills (2)

### Theme Factory
**Location:** `.claude/skills/theme-factory/`

**Available Themes (10):**
1. **Arctic Frost** - Cool blues and whites
2. **Botanical Garden** - Natural greens
3. **Desert Rose** - Warm earth tones
4. **Forest Canopy** - Deep greens and browns
5. **Golden Hour** - Warm sunset colors
6. **Midnight Galaxy** - Deep purples and blues
7. **Modern Minimalist** - Clean blacks and whites
8. **Ocean Depths** - Blues and teals
9. **Sunset Boulevard** - Oranges and purples
10. **Tech Innovation** - Cyans and grays

**Usage:**
```typescript
// Apply theme to ZELDA UI
import { applyTheme } from '@/lib/theme-factory';
applyTheme('midnight-galaxy');
```

### Internal Communications
**Location:** `.claude/skills/internal-comms/`

**Templates (4):**
1. **3P Updates** - Third-party service updates
2. **Company Newsletter** - Internal newsletters
3. **FAQ Answers** - Standardized FAQ responses
4. **General Comms** - General internal communications

**Usage:**
Ask Claude Code to generate internal communications using these templates.

---

## 📋 Configuration Files

### Hook Configuration
**File:** `.claude/settings.local.json` (gitignored - personal settings)
- Contains all 27 hook configurations
- Triggers on tool events (pre/post)
- Customizable per developer

### MCP Configuration
**File:** `.mcp.json` (committed - shared config)
- Contains all 34 MCP server definitions
- Requires environment variables for sensitive data
- Team-shared configuration

### Hook Scripts
**Location:** `.claude/hooks/`
- `conventional-commits.py` - Validates commit message format
- `prevent-direct-push.py` - Blocks direct pushes to protected branches
- `validate-branch-name.py` - Enforces branch naming

---

## 🚀 ZELDA-Specific Use Cases

### 1. Automated Testing Pipeline
**Hooks Used:**
- `test-runner` - Runs test suite
- `run-tests-after-changes` - Triggers tests on code changes
- `simple-notifications` - Notifies on test results

### 2. Database Operations
**MCP Servers Used:**
- `supabase` - Query ZELDA database
- `postgresql` - Direct PostgreSQL access
- `mysql` - MySQL connections if needed

### 3. Frontend Development
**Hooks Used:**
- `nextjs-code-quality-enforcer` - Next.js best practices
- `format-javascript-files` - Auto-format React/TS
- `lint-on-save` - Real-time linting
- `build-on-change` - Auto-rebuild on changes

### 4. Git Workflow
**Hooks Used:**
- `smart-commit` - AI-generated commit messages
- `conventional-commits` - Enforces commit format
- `auto-git-add` - Smart file staging
- `prevent-direct-push` - Protects main branch

### 5. Security & Compliance
**Hooks Used:**
- `security-scanner` - Vulnerability scanning
- `file-protection` - Protects sensitive files
- `dependency-checker` - Outdated dependency alerts

**MCP Servers Used:**
- `codacy` - Code quality metrics
- `dynatrace` - Security monitoring

### 6. Deployment Automation
**Hooks Used:**
- `deployment-health-monitor` - Monitor deploys
- `vercel-environment-sync` - Sync env vars
- `build-on-change` - Auto-build

**MCP Servers Used:**
- `pulumi` - Infrastructure deployment
- `azure-kubernetes-service` - K8s deployments

### 7. Browser Testing
**MCP Servers Used:**
- `playwright-server` - E2E testing
- `chrome-devtools` - Debugging
- `browserbase` - Cloud testing

### 8. Code Analysis
**MCP Servers Used:**
- `DeepGraph React` - Analyze React components
- `DeepGraph Next.js` - Analyze Next.js structure
- `DeepGraph TypeScript` - Analyze TypeScript code
- `context7` - Cross-codebase search

---

## 🔧 Setup & Configuration

### 1. Configure GitHub MCP
```bash
# Set your GitHub token
export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_your_token"
```

### 2. Configure Supabase MCP
```bash
# Get project ref from Supabase dashboard
# Update .mcp.json with your project-ref and token
export SUPABASE_ACCESS_TOKEN="your_token"
```

### 3. Enable Hooks
Hooks are automatically enabled via `.claude/settings.local.json`.
To disable a hook, remove it from settings or set `enabled: false`.

### 4. Test MCP Servers
```bash
# Test GitHub MCP
npx @modelcontextprotocol/server-github --help

# Test Supabase MCP
npx @supabase/mcp-server-supabase@latest --help
```

---

## 📊 Hook Event Flow

### Pre-tool Hooks (Run BEFORE tool execution)
1. `update-search-year` - Updates search context
2. `backup-before-edit` - Creates backup

### Tool Execution
*Claude Code tool runs (Read, Edit, Write, etc.)*

### Post-tool Hooks (Run AFTER tool execution)
1. `format-python-files` - Format Python
2. `format-javascript-files` - Format JS/TS
3. `git-add-changes` - Stage changes
4. `run-tests-after-changes` - Run tests
5. `simple-notifications` - Notify completion

---

## 🎯 Recommended Workflows

### Code Review Workflow
1. **Write code** → `nextjs-code-quality-enforcer` validates
2. **Save files** → `lint-on-save` + `smart-formatting` clean code
3. **Edit complete** → `format-javascript-files` formats
4. **Tests run** → `test-runner` validates
5. **Commit** → `conventional-commits` validates message
6. **Push** → `prevent-direct-push` checks branch

### Deployment Workflow
1. **Code changes** → `build-on-change` triggers build
2. **Build complete** → `simple-notifications` alerts
3. **Deploy** → `deployment-health-monitor` tracks
4. **Env sync** → `vercel-environment-sync` updates

### Security Workflow
1. **Code edit** → `security-scanner` scans
2. **Dependencies** → `dependency-checker` validates
3. **Sensitive files** → `file-protection` guards
4. **Commit** → `validate-branch-name` enforces naming

---

## 📞 Support

**Hook Documentation:** `.claude/hooks/`
**Skill Documentation:** `.claude/skills/*/SKILL.md`
**MCP Documentation:** `.mcp.json` (inline comments)

**Troubleshooting:**
1. Check `.claude/settings.local.json` for hook config
2. Verify environment variables for MCP servers
3. Review hook logs in Claude Code output
4. Test MCP servers individually via command line

---

## 🔒 Security Notes

### Sensitive Data
- Never commit `.claude/settings.local.json` (gitignored)
- Never commit API keys/tokens in `.mcp.json`
- Use environment variables for all secrets

### Protected Files
The `file-protection` hook guards:
- `.env` files
- `credentials.json`
- Private keys
- Database passwords

### Git Protection
The `prevent-direct-push` hook blocks:
- Direct pushes to `main`
- Direct pushes to `master`
- Force pushes (configurable)

---

## 📈 Performance Impact

### Hook Execution Time
- Pre-tool hooks: ~50-100ms total
- Post-tool hooks: ~200-500ms total (includes formatting/testing)
- Negligible impact on development workflow

### MCP Server Overhead
- Servers run on-demand (not always active)
- Typical response time: 100-300ms
- Browser automation: 1-3s (acceptable for E2E tests)

---

**ZELDA v2.0 - Powered by 63 Automation Components**

🤖 Generated with [Claude Code](https://claude.com/claude-code)
