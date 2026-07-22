import React, { useState, useRef } from 'react';
import './TextFileUploader.css';

/**
 * TextFileUploader - .txt or .lean file upload component for prompt context
 * Reads file contents and passes them to the parent for prompt/context handling.
 */
function TextFileUploader({ 
  onFileLoaded, 
  disabled = false,
  maxSizeMB = 5,
  showCharCount = true,
  confirmIfNotEmpty = true,
  existingPromptLength = 0
}) {
  const [status, setStatus] = useState(null); // null | { type: 'success'|'error', message: string, filename?: string }
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef(null);
  const statusTimerRef = useRef(null);  // Use ref instead of state (no re-renders needed)

  const handleButtonClick = () => {
    if (disabled || isLoading) return;
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    // Reset file input to allow re-uploading the same file
    event.target.value = '';

    // Clear previous status
    setStatus(null);

    for (const file of files) {
      const lowerFileName = file.name.toLowerCase();
      const isAllowedFile = lowerFileName.endsWith('.txt') || lowerFileName.endsWith('.lean');

      if (!isAllowedFile) {
        setStatus({
          type: 'error',
          message: 'Please select only .txt or .lean files'
        });
        autoHideStatus();
        return;
      }

      const fileSizeMB = file.size / (1024 * 1024);
      if (fileSizeMB > maxSizeMB) {
        setStatus({
          type: 'error',
          message: `${file.name} exceeds ${maxSizeMB}MB limit (actual: ${fileSizeMB.toFixed(2)}MB)`
        });
        autoHideStatus();
        return;
      }
    }

    // Show confirmation if prompt already has content
    if (confirmIfNotEmpty && existingPromptLength > 100) {
      const fileText = files.length === 1 ? 'the file' : `${files.length} files`;
      const confirmed = window.confirm(
        `Prompt already contains text. This will add ${fileText} as labeled context blocks at the end. Continue?`
      );
      if (!confirmed) {
        return;
      }
    }

    // Read file
    setIsLoading(true);
    try {
      const loadedFiles = [];

      for (const file of files) {
        const content = await readFileContent(file);

        // Strip BOM if present (UTF-8 BOM: 0xEF 0xBB 0xBF)
        const cleanContent = content.charCodeAt(0) === 0xFEFF ? content.slice(1) : content;

        if (!cleanContent.trim()) {
          setStatus({
            type: 'error',
            message: `${file.name} is empty or contains only whitespace`
          });
          autoHideStatus();
          return;
        }

        loadedFiles.push({ name: file.name, content: cleanContent });
      }

      const addedChars = loadedFiles.reduce((total, item) => total + item.content.length, 0);
      const totalChars = existingPromptLength + addedChars;

      for (const item of loadedFiles) {
        onFileLoaded(item.content, item.name);
      }

      // Show success message
      const charCountMsg = showCharCount 
        ? ` (Added: ${addedChars.toLocaleString()} chars, Total: ${totalChars.toLocaleString()})`
        : '';
      
      const warningMsg = totalChars > 50000 
        ? ' ⚠️ Large prompt may affect performance'
        : '';

      setStatus({
        type: 'success',
        message: `✓ ${loadedFiles.length} file${loadedFiles.length === 1 ? '' : 's'} added${charCountMsg}${warningMsg}`,
        filename: loadedFiles.map(item => item.name).join(', ')
      });

      autoHideStatus();

    } catch (error) {
      console.error('File read error:', error);
      setStatus({
        type: 'error',
        message: error.message || 'Failed to read file. Please try again.'
      });
      autoHideStatus();
    } finally {
      setIsLoading(false);
    }
  };

  const readFileContent = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const content = e.target.result;
          resolve(content);
        } catch (error) {
          reject(new Error('Failed to process file content'));
        }
      };
      
      reader.onerror = () => {
        reject(new Error('Failed to read file. Please try again.'));
      };
      
      reader.onabort = () => {
        reject(new Error('File reading was cancelled'));
      };

      try {
        reader.readAsText(file, 'UTF-8');
      } catch (error) {
        reject(new Error('Failed to read file. Invalid encoding.'));
      }
    });
  };

  const autoHideStatus = () => {
    // Clear any existing timer
    if (statusTimerRef.current) {
      clearTimeout(statusTimerRef.current);
    }
    
    // Set new timer and track it in ref
    statusTimerRef.current = setTimeout(() => {
      setStatus(null);
      statusTimerRef.current = null;
    }, 5000); // 5 seconds
  };

  const handleRemoveStatus = () => {
    // Clear timer when manually dismissing
    if (statusTimerRef.current) {
      clearTimeout(statusTimerRef.current);
      statusTimerRef.current = null;
    }
    setStatus(null);
  };

  // Cleanup effect - clear timer on unmount (runs only once)
  React.useEffect(() => {
    return () => {
      if (statusTimerRef.current) {
        clearTimeout(statusTimerRef.current);
      }
    };
  }, []);  // Empty array - runs only on mount/unmount

  return (
    <div className="text-file-uploader">
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.lean,text/plain,text/x-lean"
        multiple
        onChange={handleFileChange}
        style={{ display: 'none' }}
        aria-label="Upload .txt or .lean files to add as context"
        disabled={disabled || isLoading}
      />
      
      <button
        className="text-upload-btn"
        onClick={handleButtonClick}
        disabled={disabled || isLoading}
        type="button"
        title="Upload .txt or .lean files to add as labeled context"
      >
        {isLoading ? (
          <>
            <span className="upload-spinner">⟳</span>
            Reading file...
          </>
        ) : (
          <>
            📎 Upload .txt or .lean files
          </>
        )}
      </button>

      {status && (
        <div 
          className={`upload-status ${status.type}`}
          role="status"
          aria-live="polite"
        >
          <span>{status.message}</span>
          <button
            className="status-close-btn"
            onClick={handleRemoveStatus}
            aria-label="Dismiss message"
            type="button"
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}

export default TextFileUploader;

