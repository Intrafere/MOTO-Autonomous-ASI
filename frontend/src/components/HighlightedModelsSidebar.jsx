import React, { useState } from 'react';
import HelpTooltip from './HelpTooltip';
import ProofStrengthBadge from './ProofStrengthBadge';
import './settings-common.css';
import './autonomous/AutonomousResearch.css';

const OsTag = () => (
  <span className="os-tag-tooltip-anchor" tabIndex={0}>
    <span className="os-tag">OS</span>
    <span className="os-tag-tooltip">
      Open source — weights available on Hugging Face for local use with LM Studio.
    </span>
  </span>
);

const OauthTag = ({ stacked = false }) => (
  <span className={`oa-tag-tooltip-anchor${stacked ? ' oa-tag-tooltip-anchor--stacked' : ''}`} tabIndex={0}>
    <span className="oa-tag">OA</span>
    <span className="oa-tag-tooltip">
      This company has OAuth enabled for third-party programs, which typically allows discounted subscription-based usage for better affordability.
    </span>
  </span>
);

export default function HighlightedModelsSidebar() {
  const [showKothTooltip, setShowKothTooltip] = useState(false);

  return (
    <div className="settings-left-sidebar">
      <div className="known-models-sidebar">
        <h3 className="flex-row-center">
          <span>Highlighted Models</span>
          <HelpTooltip
            label="Learn about highlighted models"
            popupClassName="help-tooltip-popup--fixed"
            useFixedPosition
            fixedPlacement="side-right"
          >
            The models and hosts listed here are not affiliated with MOTO or Intrafere LLC. This chart reflects developer-tested configurations intended to help guide model selection. All statements regarding pricing, performance, roles, rankings, or capabilities are speculative and based on individual testing experience. Intrafere LLC and the MOTO development team make no guarantees about the accuracy of this chart. MOTO is compatible with the majority of models, including many not listed here.
          </HelpTooltip>
        </h3>
        <p className="hint-text hint-text--dim known-models-note">
          Note: Most models are compatible with MOTO.
        </p>
        <div className="models-list">
          <div className="models-podium">
            <div className="models-podium-label">Leaderboard</div>
            <div className="model-item model-item--ranked model-item--gold model-item--os">
              <OsTag />
              <div className="flex-row-center">
                <div className="model-item-name">MiniMax M3</div>
                <div
                  className="help-tooltip-anchor"
                  style={{ zIndex: 100 }}
                  aria-label="Learn about the King of the Hill ranking"
                  onMouseEnter={() => setShowKothTooltip(true)}
                  onMouseLeave={() => setShowKothTooltip(false)}
                  onFocus={() => setShowKothTooltip(true)}
                  onBlur={() => setShowKothTooltip(false)}
                  tabIndex={0}
                >
                  <div className="ranking-badge ranking-badge--gold ranking-badge--stacked">
                    <span className="ranking-badge-crown">👑</span>
                    <span className="ranking-badge-text">
                      <span>KING OF</span>
                      <span>THE HILL</span>
                    </span>
                  </div>
                  {showKothTooltip && (
                    <div
                      className="help-tooltip-popup"
                      style={{ top: 'auto', bottom: 'calc(100% + 10px)', left: 'calc(100% + 10px)', right: 'auto' }}
                    >
                      This model was chosen by the Intrafere developers as the best overall performer in the MOTO harness, optimized for cost, speed, and knowledge.
                    </div>
                  )}
                </div>
              </div>
              <div className="model-item-badge">Highly knowledgeable, affordable API cost</div>
            </div>

            <div className="model-item model-item--ranked model-item--silver model-item--oa">
              <OauthTag stacked />
              <div className="flex-row-center">
                <ProofStrengthBadge variant="leaderboard" className="ps-badge-anchor--model-only" />
                <div className="model-item-name">GPT 5.6</div>
                <div className="ranking-badge ranking-badge--silver">🥈 SILVER</div>
              </div>
              <div className="model-item-badge">Powerful and affordable OAuth</div>
            </div>

            <div className="model-item model-item--ranked model-item--bronze model-item--oa">
              <OauthTag />
              <div className="flex-row-center">
                <div className="model-item-name">Grok 4.5</div>
                <div className="ranking-badge ranking-badge--bronze">🥉 BRONZE</div>
              </div>
              <div className="model-item-badge">Powerful and affordable OAuth</div>
            </div>
          </div>

          <div className="model-item">
            <div className="model-item-name">Arcee AI's Trinity Large</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item">
            <div className="model-item-name">Amazon's Nova</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item">
            <ProofStrengthBadge variant="leaderboard" className="ps-badge-anchor--model-only" />
            <div className="model-item-name">Anthropic's Claude</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item">
            <div className="model-item-name">Cohere's Command A</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item">
            <div className="model-item-name">Mistral AI's Mistral Large</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">Meta's Llama</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item">
            <div className="model-item-name">Moonshot AI's Kimi</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item">
            <div className="model-item-name">AI21's Jamba</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">DeepSeek's DeepSeek</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item">
            <div className="model-item-name">Google's Gemini Pro</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">Google's Gemma</div>
            <div className="model-item-badge">Balanced knowledge and speed</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">Z.AI's GLM</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">Z.AI's GLM Turbo</div>
            <div className="model-item-badge">Fast validator</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">OpenAI's GPT OSS</div>
            <div className="model-item-badge">Balanced knowledge and speed</div>
          </div>

          <div className="model-item model-item--oa">
            <OauthTag />
            <div className="model-item-name">xAI's Grok</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item model-item--oa">
            <OauthTag stacked />
            <ProofStrengthBadge variant="leaderboard" className="ps-badge-anchor--model-only" />
            <div className="model-item-name">OpenAI's ChatGPT</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item model-item--oa">
            <OauthTag stacked />
            <ProofStrengthBadge variant="leaderboard" className="ps-badge-anchor--model-only" />
            <div className="model-item-name">Sakana AI's Fugu</div>
            <div className="model-item-badge">Highly knowledgeable model fusion</div>
          </div>

          <div className="model-item">
            <div className="model-item-name">Inception's Mercury</div>
            <div className="model-item-badge">Rapid knowledge</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">NVIDIA's Nemotron Super</div>
            <div className="model-item-badge">Balanced knowledge and speed</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">Nous Research's Hermes</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item">
            <div className="model-item-name">Perplexity's Sonar</div>
            <div className="model-item-badge">Native internet search capability</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">Microsoft's Phi</div>
            <div className="model-item-badge">Balanced knowledge and speed</div>
          </div>

          <div className="model-item">
            <div className="model-item-name">MiniMax's MiniMax</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">Alibaba's Qwen Coder</div>
            <div className="model-item-badge">Computer science</div>
          </div>

          <div className="model-item model-item--os">
            <OsTag />
            <div className="model-item-name">Alibaba's Qwen</div>
            <div className="model-item-badge">Highly knowledgeable</div>
          </div>
        </div>
      </div>
    </div>
  );
}
