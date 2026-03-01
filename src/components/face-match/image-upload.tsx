'use client';

import { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, X, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface ImageUploadProps {
  modality: 'face' | 'caricature';
  onImageSelect: (file: File, preview: string) => void;
  onImageClear?: () => void;
  preview?: string;
  disabled?: boolean;
  className?: string;
}

export function ImageUpload({
  modality,
  onImageSelect,
  onImageClear,
  preview,
  disabled = false,
  className,
}: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const processFile = useCallback(
    async (file: File) => {
      // 验证文件类型
      const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];
      if (!validTypes.includes(file.type)) {
        alert('请上传有效的图像文件 (JPG, PNG, BMP, WEBP)');
        return;
      }

      setIsUploading(true);

      try {
        // 创建预览
        const reader = new FileReader();
        reader.onload = (e) => {
          const previewResult = e.target?.result as string;
          onImageSelect(file, previewResult);
          setIsUploading(false);
        };
        reader.readAsDataURL(file);
      } catch (error) {
        console.error('Error processing file:', error);
        setIsUploading(false);
      }
    },
    [onImageSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      if (disabled) return;

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const file = files[0];
        await processFile(file);
      }
    },
    [disabled, processFile]
  );

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        await processFile(files[0]);
      }
    },
    [processFile]
  );

  const handleClear = () => {
    if (onImageClear) {
      onImageClear();
    }
  };

  return (
    <Card className={cn('overflow-hidden', className)}>
      <CardContent className="p-0">
        <div
          className={cn(
            'relative flex flex-col items-center justify-center min-h-[200px] md:min-h-[280px] transition-colors',
            isDragging && 'bg-primary/10 border-2 border-dashed border-primary',
            !preview && 'border-2 border-dashed border-muted-foreground/25 hover:border-muted-foreground/50'
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {preview ? (
            <div className="relative w-full h-full">
              <img
                src={preview}
                alt={`${modality} preview`}
                className="w-full h-full object-contain max-h-[280px]"
              />
              {!disabled && (
                <Button
                  variant="destructive"
                  size="icon"
                  className="absolute top-2 right-2 h-8 w-8"
                  onClick={handleClear}
                >
                  <X className="h-4 w-4" />
                </Button>
              )}
              <div className="absolute bottom-2 left-2 bg-background/80 px-2 py-1 rounded text-xs">
                {modality === 'face' ? '真实人脸' : '漫画人脸'}
              </div>
            </div>
          ) : (
            <label className="flex flex-col items-center justify-center w-full h-full cursor-pointer p-4">
              {isUploading ? (
                <Loader2 className="h-10 w-10 text-muted-foreground animate-spin" />
              ) : (
                <>
                  <div className="flex flex-col items-center gap-2">
                    <div className="p-4 rounded-full bg-muted">
                      {modality === 'face' ? (
                        <ImageIcon className="h-8 w-8 text-muted-foreground" />
                      ) : (
                        <Upload className="h-8 w-8 text-muted-foreground" />
                      )}
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-medium">
                        {modality === 'face' ? '上传真实人脸图像' : '上传漫画人脸图像'}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        拖拽或点击选择文件
                      </p>
                    </div>
                  </div>
                </>
              )}
              <input
                type="file"
                className="hidden"
                accept="image/*"
                onChange={handleFileSelect}
                disabled={disabled || isUploading}
              />
            </label>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
