'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Cpu, 
  HardDrive, 
  Server, 
  CheckCircle2, 
  XCircle, 
  Loader2,
  MemoryStick
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface SystemStatus {
  healthy: boolean;
  models: {
    faceDetector: { loaded: boolean; loading?: boolean; error?: string; device?: string };
    faceAligner: { loaded: boolean; loading?: boolean; error?: string };
    featureExtractor: { loaded: boolean; loading?: boolean; error?: string; device?: string };
    crossModalMatcher: { loaded: boolean; loading?: boolean; error?: string; device?: string };
  };
  memory: {
    total: number;
    used: number;
    free: number;
    usagePercent: number;
  };
  uptime: number;
}

interface SystemStatusProps {
  className?: string;
}

export function SystemStatus({ className }: SystemStatusProps) {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // 每30秒刷新
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/status');
      const data = await response.json();
      if (data.success) {
        setStatus(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch status:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
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

  if (!status) {
    return (
      <Card className={cn('w-full', className)}>
        <CardContent className="flex items-center justify-center py-8">
          <XCircle className="h-6 w-6 text-destructive mr-2" />
          <span className="text-muted-foreground">无法获取系统状态</span>
        </CardContent>
      </Card>
    );
  }

  const modelList = [
    { name: '人脸检测', key: 'faceDetector' as const, ...status.models.faceDetector },
    { name: '人脸对齐', key: 'faceAligner' as const, ...status.models.faceAligner },
    { name: '特征提取', key: 'featureExtractor' as const, ...status.models.featureExtractor },
    { name: '跨模态匹配', key: 'crossModalMatcher' as const, ...status.models.crossModalMatcher },
  ];

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Server className="h-5 w-5" />
            系统状态
          </CardTitle>
          <Badge variant={status.healthy ? 'default' : 'destructive'}>
            {status.healthy ? '运行正常' : '异常'}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 模型状态 */}
        <div>
          <p className="text-sm font-medium mb-2">模型状态</p>
          <div className="grid grid-cols-2 gap-2">
            {modelList.map((model) => (
              <div
                key={model.key}
                className="flex items-center justify-between p-2 rounded-lg border bg-muted/30"
              >
                <span className="text-sm">{model.name}</span>
                <div className="flex items-center gap-1">
                  {model.loading ? (
                    <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />
                  ) : model.loaded ? (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  ) : (
                    <XCircle className="h-4 w-4 text-red-500" />
                  )}
                  {model.device && (
                    <Badge variant="outline" className="text-xs">
                      {model.device}
                    </Badge>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 资源使用 */}
        <div className="grid grid-cols-2 gap-3">
          <div className="flex items-center gap-2 p-2 rounded-lg border bg-muted/30">
            <MemoryStick className="h-4 w-4 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">内存使用</p>
              <p className="text-sm font-medium">
                {formatBytes(status.memory.used)} / {formatBytes(status.memory.total)}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 p-2 rounded-lg border bg-muted/30">
            <Cpu className="h-4 w-4 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">运行时间</p>
              <p className="text-sm font-medium">{formatUptime(status.uptime)}</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
