import React, { useId } from 'react';
import './MasterProofGraphic.css';

// Release-highlights illustration of master proofs: the same Lean 4 proof system that
// proves a normal theorem in one go proves a large theorem one step at a time.
const STEPS = ['Step 1', 'Step 2', 'Step 3'];

function Arrow() {
  return <span className="mpg-arrow" aria-hidden="true">→</span>;
}

function LeanProof() {
  return <span className="mpg-box mpg-box--lean">Lean 4 proof</span>;
}

export default function MasterProofGraphic() {
  const captionId = `${useId()}-caption`;

  return (
    <figure className="mpg-graphic" aria-labelledby={captionId}>
      <div className="mpg-row">
        <span className="mpg-row__label">Normal proof</span>
        <div className="mpg-flow">
          <span className="mpg-box">Theorem</span>
          <Arrow />
          <LeanProof />
          <Arrow />
          <span className="mpg-box mpg-box--done">✓ Verified</span>
        </div>
      </div>

      <div className="mpg-row mpg-row--master">
        <span className="mpg-row__label">Master proof</span>
        <div className="mpg-flow">
          <span className="mpg-box mpg-box--big">Big theorem</span>
          <Arrow />
          <ol className="mpg-steps" aria-label="Steps of the big theorem">
            {STEPS.map((step) => (
              <li key={step} className="mpg-step">
                <span className="mpg-step__name">{step}</span>
                <LeanProof />
              </li>
            ))}
          </ol>
          <Arrow />
          <span className="mpg-box mpg-box--done">✓ Verified</span>
        </div>
      </div>

      <figcaption id={captionId} className="mpg-caption">
        A master proof uses the same Lean 4 proof system. It breaks a big theorem into steps and proves
        each one the normal way.
      </figcaption>
    </figure>
  );
}
