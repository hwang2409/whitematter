"use client";
import Link from "next/link";
import "./LandingPage.css";

const ARCH_LAYERS = [
  "Conv2d(1→32)", "BatchNorm", "ReLU", "MaxPool",
  "Conv2d(32→64)", "BatchNorm", "ReLU", "MaxPool",
  "Linear(1568→128)", "Linear(128→10)",
];

const EPOCHS = [
  { epoch: 6, loss: "0.082", acc: "97.3%" },
  { epoch: 7, loss: "0.061", acc: "98.0%" },
  { epoch: 8, loss: "0.048", acc: "98.5%", active: true },
];

const PREDICTIONS = [
  { label: "0", value: "0.3%" },
  { label: "1", value: "1.2%" },
  { label: "3", value: "0.8%" },
  { label: "7", value: "93.2%", highlight: true },
  { label: "8", value: "2.1%" },
  { label: "9", value: "1.4%" },
];

const HOW_STEPS = [
  {
    title: "Describe your problem",
    desc: "\u201CI have product photos and want to sort them into categories\u201D \u2014 just talk to it like you\u2019d talk to a coworker.",
  },
  {
    title: "Review the architecture",
    desc: "The AI suggests a neural network, shows you the layers and estimated accuracy. Tweak it or just hit train.",
  },
  {
    title: "Train and use",
    desc: "Training runs on our cloud. Results stream live in the chat. Test your model instantly or deploy it as an API.",
  },
];

export default function LandingPage() {
  return (
    <div className="landing">
      {/* Nav */}
      <div className="nav-wrap">
        <nav className="nav">
          <Link href="#" className="nav-logo">whitematter</Link>
          <a href="#features" className="nav-link">Features</a>
          <a href="#" className="nav-link">Docs</a>
          <a href="#" className="nav-link">Changelog</a>
          <Link href="/login" className="nav-cta">Sign In</Link>
        </nav>
      </div>

      {/* Hero */}
      <section className="hero">
        <div className="hero-kicker">ML Training Platform</div>
        <h1>Build Neural Networks Through <em>Conversation</em></h1>
        <p>
          Describe what you want to build. We design the architecture, train the
          model, and give you an API. No ML expertise required.
        </p>
        <div className="hero-buttons">
          <Link href="/register" className="btn-primary">Get Started Free →</Link>
          <a href="#features" className="btn-secondary">See How It Works</a>
        </div>
        <div className="hero-note">No credit card required. Train your first model in 30 seconds.</div>
      </section>

      {/* Product Preview */}
      <div className="preview-wrap">
        <div className="preview">
          <div className="preview-bar">
            <span className="preview-dot" />
            <span className="preview-dot" />
            <span className="preview-dot" />
          </div>
          <div className="preview-content">
            <div className="chat-msg user">
              <div className="chat-bubble user-bubble">
                I want to classify handwritten digits. I&apos;m a total beginner.
              </div>
            </div>
            <div className="chat-msg">
              <div className="chat-avatar ai">wm</div>
              <div>
                <div className="chat-bubble">
                  I&apos;ll build a <strong>CNN for MNIST</strong> — the classic
                  digit recognition dataset. This will hit{" "}
                  <strong>98.5% accuracy in about 18 seconds</strong> of training.
                </div>
                <div className="result-card">
                  <div className="result-big">98.5%</div>
                  <div className="result-label">Accuracy · trained in 18 seconds</div>
                  <div className="result-meta">
                    <span>207K parameters</span>
                    <span>10 epochs</span>
                    <span>MNIST dataset</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature 1: Chat to Build */}
      <section className="feature-section" id="features">
        <div className="feat-visual">
          <div className="feat-visual-bar"><span /><span /><span /></div>
          <div className="feat-visual-inner">
            <div style={{ fontSize: 14, color: "var(--text)", marginBottom: 14, fontWeight: 500 }}>
              MNIST Classifier — CNN
            </div>
            <div className="arch-chips">
              {ARCH_LAYERS.map((layer) => (
                <span key={layer} className="arch-chip">{layer}</span>
              ))}
            </div>
            <div className="arch-meta">207K parameters · ~18s to train · MNIST dataset</div>
          </div>
        </div>
        <div className="feat-text">
          <div className="kicker">Chat to Build</div>
          <h2>Describe Your Problem, Get a Neural Network</h2>
          <p className="lead">No forms. No configuration. Just a conversation.</p>
          <p>
            Tell WhiteMatter what you want to build in plain English. Our AI
            designs the optimal architecture — CNNs for images, Transformers for
            text, or anything in between. Review the layers, tweak if you want,
            or just hit train.
          </p>
        </div>
      </section>

      {/* Feature 2: Train in Seconds */}
      <section className="feature-section reverse">
        <div className="feat-text">
          <div className="kicker">Train in Minutes</div>
          <h2>Watch Your Model Learn in Real Time</h2>
          <p className="lead">One click to train. Live results stream into the chat.</p>
          <p>
            Training runs on our cloud — you don&apos;t need AWS, GPUs, or any
            infrastructure. Watch accuracy climb and loss drop epoch by epoch,
            with live charts right in the conversation.
          </p>
        </div>
        <div className="feat-visual">
          <div className="feat-visual-bar"><span /><span /><span /></div>
          <div className="feat-visual-inner">
            <div className="train-progress">
              <div className="train-header">
                <strong>Training · Epoch 8/10</strong>
                <span style={{ color: "var(--text-secondary)" }}>80%</span>
              </div>
              <div className="train-bar"><div className="train-bar-fill" /></div>
              <div className="train-epochs">
                {EPOCHS.map((e) => (
                  <div key={e.epoch} className="train-epoch">
                    <span style={e.active ? { color: "var(--text)", fontWeight: 500 } : undefined}>
                      Epoch {e.epoch}
                    </span>
                    <span style={e.active ? { color: "var(--text)", fontWeight: 500 } : undefined}>
                      Loss: {e.loss} · Acc: {e.acc}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature 3: Test Instantly */}
      <section className="feature-section">
        <div className="feat-visual">
          <div className="feat-visual-bar"><span /><span /><span /></div>
          <div className="feat-visual-inner">
            <div className="predict-demo">
              <div className="predict-flow">
                <div className="predict-box">image.png</div>
                <div className="predict-arrow">→</div>
                <div className="predict-box" style={{ fontFamily: "monospace", fontSize: 12 }}>
                  model.forward()
                </div>
                <div className="predict-arrow">→</div>
                <div className="predict-box result">Digit 7</div>
              </div>
              <div className="predict-conf">93.2% confidence · Tested right in the chat</div>
              <div className="predict-classes">
                {PREDICTIONS.map((p) => (
                  <span key={p.label} className={`predict-class${p.highlight ? " highlight" : ""}`}>
                    {p.label}: {p.value}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
        <div className="feat-text">
          <div className="kicker">Test Instantly</div>
          <h2>Try Your Model Without Leaving the Chat</h2>
          <p className="lead">Drop an image in. Get a prediction back.</p>
          <p>
            Once training is done, test your model right there in the
            conversation. Upload a file, see the prediction with confidence
            scores for every class. No deployment, no API calls, no page
            navigation.
          </p>
        </div>
      </section>

      {/* How It Works */}
      <section className="how-section">
        <h2>How It Works</h2>
        {HOW_STEPS.map((step, i) => (
          <div key={step.title} className="how-step">
            <div className="how-num">{i + 1}</div>
            <div>
              <h4>{step.title}</h4>
              <p>{step.desc}</p>
            </div>
          </div>
        ))}
      </section>

      {/* Final CTA */}
      <div className="final-cta">
        <h2>Start Building in 30 Seconds</h2>
        <p>
          No credit card. No AWS keys. No ML expertise.<br />
          Just describe what you want and watch it train.
        </p>
        <Link href="/register" className="btn-primary">Get Started Free →</Link>
      </div>

      {/* Footer */}
      <footer>
        <span>© 2026 WhiteMatter</span>
        <div className="footer-links">
          <a href="#">Terms</a>
          <a href="#">Privacy</a>
          <a href="#">Docs</a>
          <a href="#">Changelog</a>
        </div>
      </footer>
    </div>
  );
}
