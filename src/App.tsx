import React, { useRef, useState, useEffect } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import { Camera, Image as ImageIcon, Loader2, RefreshCw } from 'lucide-react';

interface Prediction {
  className: string;
  probability: number;
}

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const modelRef = useRef<mobilenet.MobileNet | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const retryCount = useRef(0);

  const loadModel = async () => {
    try {
      setIsModelLoading(true);
      setError(null);
      
      // Ensure TensorFlow.js is properly initialized
      await tf.ready();
      
      // Set backend to 'webgl' for better performance
      await tf.setBackend('webgl');
      
      const loadedModel = await mobilenet.load({
        version: 2,
        alpha: 1.0
      });
      
      modelRef.current = loadedModel;
      setIsModelLoading(false);
      retryCount.current = 0;
    } catch (err) {
      console.error('Model loading error:', err);
      if (retryCount.current < 3) {
        retryCount.current += 1;
        setError(`Loading failed. Retrying... (Attempt ${retryCount.current}/3)`);
        setTimeout(loadModel, 2000); // Retry after 2 seconds
      } else {
        setError('Unable to load AI model. Please check your internet connection and refresh the page.');
        setIsModelLoading(false);
      }
    }
  };

  useEffect(() => {
    let mounted = true;

    const initializeModel = async () => {
      if (mounted) {
        await loadModel();
      }
    };

    initializeModel();

    return () => {
      mounted = false;
    };
  }, []);

  const handleRetry = () => {
    retryCount.current = 0;
    loadModel();
  };

  const analyzeImage = async (imageElement: HTMLImageElement) => {
    if (!modelRef.current) {
      setError('Model not loaded yet. Please try again in a moment.');
      return;
    }

    try {
      setIsAnalyzing(true);
      setError(null);
      
      // Ensure the image is loaded before analysis
      await new Promise((resolve) => {
        if (imageElement.complete) resolve(true);
        imageElement.onload = () => resolve(true);
      });

      const predictions = await modelRef.current.classify(imageElement, 5);
      setPredictions(predictions);
    } catch (err) {
      setError('Error analyzing image. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPG, PNG)');
      return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
      const dataUrl = e.target?.result as string;
      setImage(dataUrl);
      setPredictions([]);

      const img = new Image();
      img.onload = () => analyzeImage(img);
      img.onerror = () => setError('Error loading image. Please try another file.');
      img.src = dataUrl;
    };
    reader.onerror = () => setError('Error reading file. Please try again.');
    reader.readAsDataURL(file);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const dataUrl = e.target?.result as string;
        setImage(dataUrl);
        setPredictions([]);

        const img = new Image();
        img.onload = () => analyzeImage(img);
        img.onerror = () => setError('Error loading image. Please try another file.');
        img.src = dataUrl;
      };
      reader.onerror = () => setError('Error reading file. Please try again.');
      reader.readAsDataURL(file);
    } else {
      setError('Please upload an image file (JPG, PNG)');
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  if (isModelLoading || error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 p-8 text-gray-100 flex items-center justify-center">
        <div className="text-center">
          {isModelLoading ? (
            <>
              <Loader2 className="w-12 h-12 mx-auto animate-spin text-indigo-400 mb-4" />
              <h2 className="text-xl font-semibold text-white mb-2">Загрузка модели</h2>
              <p className="text-gray-400">Пожалуйста, подождите, пока мы инициализируем приложение...</p>
            </>
          ) : (
            <>
              <div className="bg-red-900/50 text-red-300 p-4 rounded-lg mb-4 border border-red-800">
                {error}
              </div>
              <button
                onClick={handleRetry}
                className="inline-flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Retry Loading
              </button>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 p-8 text-gray-100">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Анализ изображений</h1>
          <p className="text-gray-400">Загрузите изображение, чтобы проанализировать его содержимое с помощью ИИ</p>
        </div>

        <div
          className="bg-gray-800 rounded-xl shadow-lg p-8 mb-8 border border-gray-700"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          {!image ? (
            <div
              className="border-2 border-dashed border-gray-600 rounded-lg p-12 text-center cursor-pointer hover:border-indigo-500 transition-colors"
              onClick={() => fileInputRef.current?.click()}
            >
              <Camera className="w-12 h-12 mx-auto mb-4 text-gray-500" />
              <p className="text-gray-400 mb-2">Перетащите изображение сюда или щелкните, чтобы выбрать</p>
              <p className="text-sm text-gray-500">Поддерживает файлы JPG, PNG</p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="relative">
                <img
                  src={image}
                  alt="Uploaded"
                  className="w-full h-64 object-cover rounded-lg"
                />
                <button
                  onClick={() => {
                    setImage(null);
                    setPredictions([]);
                    setError(null);
                  }}
                  className="absolute top-2 right-2 bg-gray-800/80 p-2 rounded-full hover:bg-gray-700 transition-colors"
                >
                  <ImageIcon className="w-5 h-5 text-gray-300" />
                </button>
              </div>
            </div>
          )}
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/*"
            onChange={handleImageUpload}
          />
        </div>

        {isAnalyzing && (
          <div className="text-center p-4">
            <Loader2 className="w-8 h-8 mx-auto animate-spin text-indigo-400" />
            <p className="text-gray-400 mt-2">Анализ изображения...</p>
          </div>
        )}

        {error && (
          <div className="bg-red-900/50 text-red-300 p-4 rounded-lg mb-8 border border-red-800">
            {error}
          </div>
        )}

        {predictions.length > 0 && (
          <div className="bg-gray-800 rounded-xl shadow-lg p-8 border border-gray-700">
            <h2 className="text-2xl font-semibold mb-4 text-white">Результаты анализа</h2>
            <div className="space-y-4">
              {predictions.map((prediction, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 bg-gray-700/50 rounded-lg"
                >
                  <span className="text-gray-200 font-medium">
                    {prediction.className}
                  </span>
                  <span className="text-indigo-400 font-semibold">
                    {(prediction.probability * 100).toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;