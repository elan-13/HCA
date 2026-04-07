function UploadImage({ onFileSelect, selectedFile, loading, onSubmit }) {
  const handleChange = (event) => {
    const file = event.target.files?.[0] || null
    onFileSelect(file)
  }

  return (
    <div className="card">
      <h2>Upload Skin Image</h2>
      <p className="subtext">Choose a clear photo, then run prediction.</p>

      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
        disabled={loading}
      />

      <button
        type="button"
        onClick={onSubmit}
        disabled={!selectedFile || loading}
      >
        {loading ? 'Predicting...' : 'Predict'}
      </button>
    </div>
  )
}

export default UploadImage
