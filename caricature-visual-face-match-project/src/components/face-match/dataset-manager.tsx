'use client';

import { useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Image as ImageIcon, 
  Upload, 
  Trash2, 
  RefreshCw,
  Loader2,
  FolderOpen
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface DatasetInfo {
  count: number;
  files: {
    name: string;
    path: string;
    size: number;
    modified: string;
  }[];
  totalSize: number;
}

interface DatasetManagerProps {
  onRefresh?: () => void;
  className?: string;
}

export function DatasetManager({ onRefresh, className }: DatasetManagerProps) {
  const [facesData, setFacesData] = useState<DatasetInfo | null>(null);
  const [caricaturesData, setCaricaturesData] = useState<DatasetInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [activeTab, setActiveTab] = useState('faces');

  const fetchDataset = useCallback(async () => {
    try {
      const response = await fetch('/api/dataset');
      const data = await response.json();
      if (data.success) {
        setFacesData(data.data.faces);
        setCaricaturesData(data.data.caricatures);
      }
    } catch (error) {
      console.error('Failed to fetch dataset:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDataset();
  }, [fetchDataset]);

  const handleUpload = async (modality: 'face' | 'caricature', files: FileList) => {
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('modality', modality);
      Array.from(files).forEach((file) => {
        formData.append('files', file);
      });

      const response = await fetch('/api/upload', {
        method: 'PUT',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        await fetchDataset();
        onRefresh?.();
      }
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (paths: string[]) => {
    try {
      const response = await fetch('/api/dataset', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files: paths }),
      });

      const data = await response.json();
      if (data.success) {
        await fetchDataset();
        onRefresh?.();
      }
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const renderDatasetContent = (data: DatasetInfo | null, modality: 'face' | 'caricature') => {
    if (!data) return null;

    return (
      <div className="space-y-3">
        {/* 统计信息 */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
          <div className="flex items-center gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold">{data.count}</p>
              <p className="text-xs text-muted-foreground">图像数量</p>
            </div>
            <div className="w-px h-8 bg-border" />
            <div className="text-center">
              <p className="text-lg font-medium">{formatBytes(data.totalSize)}</p>
              <p className="text-xs text-muted-foreground">总大小</p>
            </div>
          </div>
          <div className="flex gap-2">
            <label>
              <input
                type="file"
                multiple
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files) {
                    handleUpload(modality, e.target.files);
                  }
                }}
                disabled={uploading}
              />
              <Button variant="outline" size="sm" asChild disabled={uploading}>
                <span className="cursor-pointer">
                  {uploading ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-1" />
                  ) : (
                    <Upload className="h-4 w-4 mr-1" />
                  )}
                  上传
                </span>
              </Button>
            </label>
            <Button
              variant="outline"
              size="sm"
              onClick={() => fetchDataset()}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* 文件列表 */}
        <div className="max-h-[300px] overflow-y-auto space-y-1">
          {data.files.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <FolderOpen className="h-10 w-10 text-muted-foreground/50 mb-2" />
              <p className="text-sm text-muted-foreground">暂无图像</p>
              <p className="text-xs text-muted-foreground">点击上传按钮添加图像</p>
            </div>
          ) : (
            data.files.map((file) => (
              <div
                key={file.path}
                className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <ImageIcon className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                  <span className="text-sm truncate">{file.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">
                    {formatBytes(file.size)}
                  </Badge>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => handleDelete([file.path])}
                  >
                    <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <Card className={cn('w-full', className)}>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">数据集管理</CardTitle>
        <CardDescription>管理测试图像数据集</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="faces">
              真实人脸 ({facesData?.count || 0})
            </TabsTrigger>
            <TabsTrigger value="caricatures">
              漫画人脸 ({caricaturesData?.count || 0})
            </TabsTrigger>
          </TabsList>
          <TabsContent value="faces" className="mt-3">
            {renderDatasetContent(facesData, 'face')}
          </TabsContent>
          <TabsContent value="caricatures" className="mt-3">
            {renderDatasetContent(caricaturesData, 'caricature')}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
