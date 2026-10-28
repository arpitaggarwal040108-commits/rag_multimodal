import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  BookOpen,
  UploadCloud,
  Files,
  Trash2,
  LoaderCircle,
  FileText,
  Send,
  Sparkles,
  ExternalLink,
  Database,
  Search,
  AlertCircle,
  Plus,
} from 'lucide-react';

const API =
  (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

async function readApiResponse(response) {
  const raw = await response.text();
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return { detail: raw.slice(0, 500) || `HTTP ${response.status}` };
  }
}

export default function App() {
  const [papers, setPapers] = useState([]);
  const [paperStats, setPaperStats] = useState(null);
  const [papersLoading, setPapersLoading] = useState(true);
  const [papersError, setPapersError] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [queryError, setQueryError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [notice, setNotice] = useState('');
  const [activeTab, setActiveTab] = useState('research');
  const [workspaces, setWorkspaces] = useState([{ id: 'default', name: 'General Research' }]);
  const [workspaceId, setWorkspaceId] = useState('default');
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [deletingPaper, setDeletingPaper] = useState('');

  const inputRef = useRef(null);

  async function refreshPapers(targetWorkspaceId = workspaceId) {
    setPapersLoading(true);
    setPapersError('');

    try {
      const response = await fetch(`${API}/api/papers?workspace_id=${encodeURIComponent(targetWorkspaceId)}`);
      if (!response.ok) throw new Error('Failed to load papers.');

      const data = await readApiResponse(response);
      setPapers(data.papers || []);
      setPaperStats({
        paperCount: data.paper_count,
        totalChunks: data.total_chunks,
      });
    } catch (error) {
      const message =
        error.message || 'Cannot connect to the research library.';
      setPapersError(message);
      setNotice(
        'Cannot connect to API. Start the FastAPI backend on port 8000.'
      );
    } finally {
      setPapersLoading(false);
    }
  }

  async function refreshWorkspaces() {
    try {
      const response = await fetch(`${API}/api/workspaces`);
      const data = await readApiResponse(response);
      if (!response.ok) throw new Error(data.detail || 'Failed to load workspaces.');
      const items = data.workspaces || [];
      setWorkspaces(items.length ? items : [{ id: 'default', name: 'General Research' }]);
      if (items.length && !items.some((item) => item.id === workspaceId)) {
        setWorkspaceId(items[0].id);
        await refreshPapers(items[0].id);
      }
    } catch (error) {
      setNotice(error.message || 'Could not load workspaces.');
    }
  }

  useEffect(() => {
    refreshWorkspaces();
    refreshPapers('default');
  }, []);

  async function changeWorkspace(nextId) {
    setWorkspaceId(nextId);
    setAnswer(null);
    setQueryError('');
    await refreshPapers(nextId);
  }

  async function createWorkspace() {
    const name = window.prompt('Name this research workspace:');
    if (!name || !name.trim()) return;
    setCreatingWorkspace(true);
    try {
      const response = await fetch(`${API}/api/workspaces`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim() }),
      });
      const data = await readApiResponse(response);
      if (!response.ok) throw new Error(data.detail || 'Could not create workspace.');
      await refreshWorkspaces();
      await changeWorkspace(data.workspace.id);
      setNotice(`Created workspace: ${data.workspace.name}`);
    } catch (error) {
      setNotice(error.message || 'Could not create workspace.');
    } finally {
      setCreatingWorkspace(false);
    }
  }

  async function uploadFiles(fileList) {
    if (!fileList?.length) return;

    setUploading(true);
    setNotice('');

    try {
      for (const file of Array.from(fileList)) {
        const form = new FormData();
        form.append('file', file);
        form.append('workspace_id', workspaceId);

        const response = await fetch(`${API}/api/upload`, {
          method: 'POST',
          body: form,
        });

        const data = await readApiResponse(response);

        if (!response.ok) {
          throw new Error(
            data.detail || `Upload failed for ${file.name}`
          );
        }

        setNotice(
          `Indexed ${data.filename}: ${data.chunks_created} chunks created.`
        );
      }

      await refreshPapers();
    } catch (error) {
      setNotice(error.message || 'Could not upload the selected files.');
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  }

  async function askQuestion(event) {
    event.preventDefault();

    if (!question.trim()) return;

    setAsking(true);
    setNotice('');
    setAnswer(null);
    setQueryError('');

    try {
      const response = await fetch(`${API}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          top_k: 5,
          workspace_id: workspaceId,
        }),
      });

      const data = await readApiResponse(response);

      if (!response.ok) {
        throw new Error(data.detail || 'Query failed.');
      }

      setAnswer(data);
    } catch (error) {
      const message = error.message || 'Could not process your question.';
      setQueryError(message);
      setNotice(message);
    } finally {
      setAsking(false);
    }
  }

  async function removePaper(paper) {
    const displayName = paper.original_filename || paper.filename;
    if (!window.confirm(`Delete "${displayName}" and all of its indexed passages?`)) return;
    setDeletingPaper(paper.filename);
    setNotice('');
    try {
      const response = await fetch(
        `${API}/api/papers/${encodeURIComponent(paper.filename)}?workspace_id=${encodeURIComponent(workspaceId)}`,
        { method: 'DELETE' },
      );
      const data = await readApiResponse(response);
      if (!response.ok) throw new Error(data.detail || 'Could not delete paper.');
      setAnswer(null);
      await refreshPapers(workspaceId);
      await refreshWorkspaces();
      setNotice(data.message || `Deleted ${displayName}.`);
    } catch (error) {
      setNotice(error.message || 'Could not delete paper.');
    } finally {
      setDeletingPaper('');
    }
  }

  async function clearLibrary() {
    if (
      !window.confirm(
        'Delete all uploaded papers and indexed chunks? This cannot be undone.'
      )
    ) {
      return;
    }

    setClearing(true);
    setNotice('');

    try {
      const response = await fetch(`${API}/api/clear?workspace_id=${encodeURIComponent(workspaceId)}`, {
        method: 'DELETE',
      });

      const data = await readApiResponse(response);

      if (!response.ok) {
        throw new Error(data.detail || 'Could not clear the library.');
      }

      setPapers([]);
      setPaperStats({ paperCount: 0, totalChunks: 0 });
      setAnswer(null);
      setQueryError('');
      setNotice(data.message || 'Library cleared.');
      await refreshWorkspaces();
    } catch (error) {
      setNotice(error.message || 'Could not clear the library.');
    } finally {
      setClearing(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
  <img src="/insightforge-logo.svg" alt="Research Prism logo" />
</span>
          <div>
            <b>Research Prism</b>
            <small>RESEARCH WORKSPACE</small>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <button
          type="button"
          className={`nav-item ${
            activeTab === 'research' ? 'active' : ''
          }`}
          onClick={() => setActiveTab('research')}
        >
          <Sparkles size={22} />
          <span>Research Assistant</span>
        </button>

        <button
          type="button"
          className={`nav-item paper-library-btn ${
            activeTab === 'library' ? 'active' : ''
          }`}
          onClick={() => setActiveTab('library')}
        >
          <Files size={22} />
          <span>Paper Library</span>
          <span className="nav-count">{papers.length}</span>
        </button>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          Local RAG engine
          <span className="online">READY</span>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <span className="eyebrow">HYBRID MULTIMODAL RAG</span>
            <h1>
              {activeTab === 'library'
                ? 'Paper Library'
                : 'Research Assistant'}
            </h1>
            <p>
              {activeTab === 'library'
                ? 'Manage your indexed research papers.'
                : 'From papers to verified insights.'}
            </p>
          </div>

          <div className="top-pill" style={{ gap: 8, flexWrap: 'wrap' }}>
            <label htmlFor="workspace-select" style={{ fontSize: 12 }}>Workspace</label>
            <select
              id="workspace-select"
              value={workspaceId}
              onChange={(event) => changeWorkspace(event.target.value)}
              disabled={papersLoading || creatingWorkspace}
              style={{ maxWidth: 190, padding: '7px 9px', borderRadius: 8, border: '1px solid var(--border, #d9dce7)', background: 'var(--panel, #fff)', color: 'inherit' }}
            >
              {workspaces.map((workspace) => (
                <option key={workspace.id} value={workspace.id}>{workspace.name}</option>
              ))}
            </select>
            <button type="button" className="icon-button" title="Create workspace" aria-label="Create workspace" onClick={createWorkspace} disabled={creatingWorkspace}>
              {creatingWorkspace ? <LoaderCircle size={15} className="spin" /> : <Plus size={16} />}
            </button>
          </div>
        </header>

        {notice && (
          <div className="notice" role="status">
            {notice}
          </div>
        )}

        {activeTab === 'research' && (
          <>
            <section className="welcome">
              <div>
                <div className="welcome-kicker">
                  <Sparkles size={15} />
                  YOUR RESEARCH COPILOT
                </div>

                <h2>
                  Ask better questions.
                  <br />
                  <em>Discover stronger evidence.</em>
                </h2>

                <p>
                  Upload research papers and ask questions across your
                  library. Every response includes supporting passages
                  and source references.
                </p>
              </div>

              <div className="welcome-art">
                <BookOpen size={78} strokeWidth={1.1} />
                <span className="orb orb-one" />
                <span className="orb orb-two" />
              </div>
            </section>

            <div className="content-grid research-grid">
              <section className="panel query-panel">
                <div className="panel-head">
                  <div>
                    <span className="step">01</span>
                    <h3>Ask your research question</h3>
                  </div>
                  <span className="method-badge">
                    <span className="status-dot" />
                    Hybrid retrieval
                  </span>
                </div>

                <form onSubmit={askQuestion}>
                  <label htmlFor="question">
                    What would you like to investigate?
                  </label>

                  <textarea
                    id="question"
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Compare the methodologies used across these papers…"
                    rows={7}
                  />

                  <div className="query-foot">
                    <span>
                      {papersLoading
                        ? 'Checking your indexed library…'
                        : `${paperStats?.paperCount ?? papers.length} indexed ${
                            (paperStats?.paperCount ?? papers.length) === 1
                              ? 'paper'
                              : 'papers'
                          }`}
                    </span>

                    <button
                      className="primary"
                      type="submit"
                      disabled={asking || question.trim().length < 3}
                    >
                      {asking ? (
                        <>
                          <LoaderCircle size={16} className="spin" />
                          Researching
                        </>
                      ) : (
                        <>
                          <Send size={16} />
                          Generate answer
                        </>
                      )}
                    </button>
                  </div>
                </form>

                <div className="retrieval-flow">
                  <div>
                    <span className="flow-number">A</span>
                    <b>MiniLM</b>
                    <small>Text relevance</small>
                  </div>

                  <span className="flow-plus">+</span>

                  <div>
                    <span className="flow-number violet">B</span>
                    <b>CLIP</b>
                    <small>Semantic signal</small>
                  </div>

                  <span className="flow-arrow">→</span>

                  <div>
                    <span className="flow-number green">R</span>
                    <b>Rerank</b>
                    <small>RRF + cosine</small>
                  </div>
                </div>

                <div className="suggestions">
                  <span>TRY A QUESTION</span>
                  {[
                    'What are the main findings across these papers?',
                    'Compare the methods used in this research.',
                  ].map((suggestion) => (
                    <button
                      type="button"
                      key={suggestion}
                      onClick={() => setQuestion(suggestion)}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </section>

              <section className="panel results-panel" aria-live="polite">
                {asking ? (
                  <div className="results-state">
                    <span className="state-icon loading-icon">
                      <LoaderCircle className="spin" size={22} />
                    </span>
                    <h3>Searching your research library</h3>
                    <p>
                      Finding relevant passages and preparing source references.
                    </p>
                    <div className="loading-bar"><span /></div>
                  </div>
                ) : queryError ? (
                  <div className="results-state error-state" role="alert">
                    <span className="state-icon error-icon">
                      <AlertCircle size={21} />
                    </span>
                    <h3>We couldn’t complete that search</h3>
                    <p>{queryError}</p>
                    {!papersLoading && papers.length === 0 && (
                      <button
                        type="button"
                        className="secondary-action"
                        onClick={() => setActiveTab('library')}
                      >
                        <Files size={15} />
                        Open Paper Library
                      </button>
                    )}
                  </div>
                ) : answer ? (
                  <div className="answer-block">
                    <div className="results-heading">
                      <div>
                        <span className="results-kicker">RESEARCH RESULTS</span>
                        <div className="answer-title">
                          <Sparkles size={17} />
                          <h3>Evidence synthesis</h3>
                        </div>
                      </div>
                      <span className="result-count">
                        {answer.retrieved_chunks ?? answer.sources?.length ?? 0}{' '}
                        passages retrieved
                      </span>
                    </div>

                    <div className="answer-text">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          a: ({ href, children }) => (
                            <a href={href} target="_blank" rel="noopener noreferrer">
                              {children}
                            </a>
                          ),
                          table: ({ children }) => (
                            <div className="markdown-table-wrapper">
                              <table>{children}</table>
                            </div>
                          ),
                        }}
                      >
                        {answer.answer || 'No answer was generated.'}
                      </ReactMarkdown>
                    </div>

                    {Array.isArray(answer.visual_results) && answer.visual_results.length > 0 && (
                      <section className="visual-evidence" aria-label="Retrieved visual evidence">
                        <div className="section-heading">
                          <div>
                            <span className="results-kicker">FIGURES & VISUALS</span>
                            <h4>Related visual evidence</h4>
                          </div>
                          <span className="section-count">{answer.visual_results.length}</span>
                        </div>
                        <div className="visual-results-grid">
                          {answer.visual_results.map((visual, index) => {
                            const rawUrl = visual.url || '';
                            const imageUrl = rawUrl && (/^https?:\/\//i.test(rawUrl) ? rawUrl : `${API}${rawUrl.startsWith('/') ? '' : '/'}${rawUrl}`);
                            const sourceName = visual.original_filename || visual.source || 'Research document';
                            return (
                              <article className="visual-result-card" key={`${sourceName}-${visual.page ?? 'page'}-${index}`}>
                                <a className="visual-preview" href={imageUrl || undefined} target="_blank" rel="noopener noreferrer" aria-label={`Open visual from ${sourceName}`}>
                                  {imageUrl ? <img src={imageUrl} alt={`Retrieved visual from ${sourceName}${visual.page ? `, page ${visual.page}` : ''}`} loading="lazy" /> : <span className="visual-placeholder">Preview unavailable</span>}
                                </a>
                                <div className="visual-result-info">
                                  <b title={sourceName}>{sourceName}</b>
                                  <small>{visual.page != null ? `Page ${visual.page}` : 'Page not available'}</small>
                                </div>
                              </article>
                            );
                          })}
                        </div>
                      </section>
                    )}

                    <div className="sources-title">
                      SUPPORTING PASSAGES
                      <span>{answer.sources?.length || 0}</span>
                    </div>

                    {(answer.sources || []).length ? (
                      answer.sources.map((source, index) => {
                        const page =
                          source.page ??
                          source.page_number ??
                          [
                            ...new Set(
                              Array.from(
                                source.text?.matchAll(/\[Page\s+(\d+)\]/gi) ||
                                  [],
                                (match) => match[1]
                              )
                            ),
                          ];
                        const pageLabel =
                          page == null
                            ? ''
                            : Array.isArray(page)
                              ? page.join(', ')
                              : String(page);
                        return (
                          <details
                            className="source-card"
                            key={`${source.source}-${source.chunk_id}-${index}`}
                          >
                            <summary>
                              <span className="source-number">
                                {String(index + 1).padStart(2, '0')}
                              </span>

                              <span>
                                <b>{source.source || 'Source document'}</b>
                                <small>
                                  {pageLabel
                                    ? `${pageLabel.includes(',') ? 'Pages' : 'Page'} ${pageLabel} · `
                                    : ''}
                                  Chunk{' '}
                                  {source.chunk_id ?? source.chunk_index}
                                  {source.score != null
                                    ? ` · relevance ${source.score}`
                                    : ''}
                                </small>
                              </span>

                              <ExternalLink size={15} />
                            </summary>

                            <p>{source.text}</p>
                          </details>
                        );
                      })
                    ) : (
                      <div className="empty-state compact-empty">
                        No supporting passages were returned for this query.
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="library-overview">
                    <div className="results-heading">
                      <div>
                        <span className="results-kicker">YOUR LIBRARY</span>
                        <h3>Research at a glance</h3>
                      </div>
                      <span className="overview-icon">
                        <Database size={20} />
                      </span>
                    </div>

                    {papersError ? (
                      <div className="inline-error" role="alert">
                        <AlertCircle size={17} />
                        <span>{papersError}</span>
                        <button type="button" onClick={refreshPapers}>
                          Retry
                        </button>
                      </div>
                    ) : papersLoading ? (
                      <div className="overview-loading">
                        <LoaderCircle className="spin" size={18} />
                        Loading your indexed library…
                      </div>
                    ) : (
                      <>
                        <div className="stats-grid">
                          <div className="stat-card">
                            <span>INDEXED PAPERS</span>
                            <b>{paperStats?.paperCount ?? papers.length}</b>
                            <small>Available to search</small>
                          </div>
                          <div className="stat-card">
                            <span>SEARCHABLE PASSAGES</span>
                            <b>{paperStats?.totalChunks ?? 0}</b>
                            <small>Across your library</small>
                          </div>
                        </div>

                        {papers.length ? (
                          <div className="recent-papers">
                            <div className="recent-heading">
                              <div>
                                <h4>Indexed papers</h4>
                                <span>Ready for evidence-based search</span>
                              </div>
                              <button
                                type="button"
                                onClick={() => setActiveTab('library')}
                              >
                                View library
                                <ExternalLink size={13} />
                              </button>
                            </div>
                            <div className="paper-list compact-paper-list">
                              {papers.slice(0, 4).map((paper) => (
                                <div
                                  className="paper-row"
                                  key={paper.filename}
                                >
                                  <span className="file-icon">
                                    <FileText size={17} />
                                  </span>
                                  <div className="paper-info">
                                    <b title={paper.original_filename || paper.filename}>
                                      {paper.original_filename || paper.filename}
                                    </b>
                                    <small>
                                      {paper.chunks} searchable passages
                                    </small>
                                  </div>
                                  <span className="indexed-dot" />
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : (
                          <div className="empty-state">
                            <span className="state-icon">
                              <Search size={20} />
                            </span>
                            <h4>No indexed papers yet</h4>
                            <p>
                              Add a paper to build your searchable research
                              library. Your answers will be grounded in those
                              sources.
                            </p>
                            <button
                              type="button"
                              className="secondary-action"
                              onClick={() => setActiveTab('library')}
                            >
                              <UploadCloud size={15} />
                              Add your first paper
                            </button>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </section>
            </div>
          </>
        )}

        {activeTab === 'library' && (
          <div className="content-grid library-view">
            <section className="panel upload-panel">
              <div className="panel-head">
                <div>
                  <span className="step">01</span>
                  <h3>Build your library</h3>
                </div>
                <span className="muted">PDF · Images · TXT · MD</span>
              </div>

              <button
                type="button"
                className="dropzone"
                onClick={() => inputRef.current?.click()}
                disabled={uploading}
              >
                <span className="upload-icon">
                  {uploading ? (
                    <LoaderCircle className="spin" size={25} />
                  ) : (
                    <UploadCloud size={25} />
                  )}
                </span>

                <b>
                  {uploading
                    ? 'Processing document…'
                    : 'Upload research papers'}
                </b>

                <span>Click to browse files · 25 MB max per file</span>
                <small>PDF, PNG, JPG, TIFF, TXT, Markdown</small>
              </button>

              <input
                ref={inputRef}
                type="file"
                multiple
                accept=".pdf,.png,.jpg,.jpeg,.bmp,.tiff,.txt,.md"
                hidden
                onChange={(event) => uploadFiles(event.target.files)}
              />

              <div className="library-head">
                <div>
                  <h4>Indexed papers</h4>
                  <span>
                    {papers.length} documents in your workspace
                  </span>
                </div>

                <button
                  type="button"
                  className="icon-button"
                  title="Clear library"
                  aria-label="Clear library"
                  onClick={clearLibrary}
                  disabled={clearing || papers.length === 0}
                >
                  {clearing ? (
                    <LoaderCircle size={16} className="spin" />
                  ) : (
                    <Trash2 size={16} />
                  )}
                </button>
              </div>

              <div className="paper-list">
                {papers.length ? (
                  papers.map((paper) => (
                    <div className="paper-row" key={paper.filename}>
                      <span className="file-icon">
                        <FileText size={18} />
                      </span>

                      <div className="paper-info">
                        <b title={paper.original_filename || paper.filename}>{paper.original_filename || paper.filename}</b>
                        <small>
                          {paper.chunks} searchable passages
                        </small>
                      </div>

                      <span className="indexed-dot" />
                      <button
                        type="button"
                        className="icon-button"
                        title={`Delete ${paper.original_filename || paper.filename}`}
                        aria-label={`Delete ${paper.original_filename || paper.filename}`}
                        onClick={() => removePaper(paper)}
                        disabled={Boolean(deletingPaper) || clearing}
                      >
                        {deletingPaper === paper.filename ? <LoaderCircle size={15} className="spin" /> : <Trash2 size={15} />}
                      </button>
                    </div>
                  ))
                ) : (
                  <div className="empty-library">
                    Your library is empty. Upload a paper to begin.
                  </div>
                )}
              </div>
            </section>
          </div>
        )}

        <footer>
          <span>RESEARCH PRISM</span>
          <span>FastAPI · FAISS · Local storage</span>
          <span>Research responsibly. Verify citations.</span>
        </footer>
      </main>
    </div>
  );
}
