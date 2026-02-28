'use client';

import { MatchedPair } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { CheckCircle2, XCircle, Trophy, ImageIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MatchResultProps {
  matches: MatchedPair[];
  queryModality: 'face' | 'caricature';
  processTime?: number;
  className?: string;
}

export function MatchResult({
  matches,
  queryModality,
  processTime,
  className,
}: MatchResultProps) {
  if (matches.length === 0) {
    return (
      <Card className={cn('w-full', className)}>
        <CardContent className="flex flex-col items-center justify-center py-12">
          <p className="text-muted-foreground">暂无匹配结果</p>
        </CardContent>
      </Card>
    );
  }

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.8) return 'text-green-600';
    if (similarity >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getSimilarityBgColor = (similarity: number) => {
    if (similarity >= 0.8) return 'bg-green-100';
    if (similarity >= 0.6) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  // Format processing time with appropriate unit
  const formatProcessTime = (ms: number): string => {
    if (ms < 1000) {
      return `${ms}ms`;
    } else if (ms < 60000) {
      return `${(ms / 1000).toFixed(2)}s`;
    } else if (ms < 3600000) {
      const minutes = Math.floor(ms / 60000);
      const seconds = ((ms % 60000) / 1000).toFixed(1);
      return `${minutes}m ${seconds}s`;
    } else {
      const hours = Math.floor(ms / 3600000);
      const minutes = Math.floor((ms % 3600000) / 60000);
      return `${hours}h ${minutes}m`;
    }
  };

  return (
    <Card className={cn('w-full flex flex-col', className)}>
      <CardHeader className="pb-2 flex-shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">匹配结果</CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              共 {matches.length} 条
            </Badge>
            {processTime !== undefined && processTime !== null && (
              <Badge variant="outline" className="text-xs">
                耗时: {formatProcessTime(processTime)}
              </Badge>
            )}
          </div>
        </div>
        <p className="text-sm text-muted-foreground">
          查询模式: {queryModality === 'face' ? '真实人脸 → 漫画' : '漫画 → 真实人脸'}
        </p>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0 min-h-0">
        <div className="h-full overflow-y-auto px-6 pb-6" style={{ maxHeight: 'calc(100vh - 400px)' }}>
          <div className="space-y-3 pt-2">
            {matches.map((match, index) => (
              <div
                key={match.imageId || `match-${index}`}
                className={cn(
                  'flex items-center gap-4 p-3 rounded-lg border transition-colors',
                  index === 0 && 'ring-2 ring-primary/50 bg-primary/5',
                  match.isMatch ? 'border-green-200 bg-green-50/50' : 'border-border'
                )}
              >
                {/* 排名 */}
                <div
                  className={cn(
                    'flex items-center justify-center w-10 h-10 rounded-full text-sm font-bold flex-shrink-0',
                    index === 0
                      ? 'bg-yellow-100 text-yellow-700'
                      : index === 1
                      ? 'bg-gray-100 text-gray-700'
                      : index === 2
                      ? 'bg-orange-100 text-orange-700'
                      : 'bg-muted text-muted-foreground'
                  )}
                >
                  {index === 0 ? (
                    <Trophy className="h-5 w-5" />
                  ) : (
                    match.rank
                  )}
                </div>

                {/* 图像预览 - 更大更清晰 */}
                <div className="w-16 h-16 rounded-lg overflow-hidden bg-muted flex items-center justify-center flex-shrink-0 border">
                  {match.imageData ? (
                    <img
                      src={`data:image/jpeg;base64,${match.imageData}`}
                      alt={`Match ${match.rank}`}
                      className="w-full h-full object-cover"
                    />
                  ) : match.imagePath ? (
                    <img
                      src={`/api/image?path=${encodeURIComponent(match.imagePath)}`}
                      alt={`Match ${match.rank}`}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <ImageIcon className="h-6 w-6 text-muted-foreground" />
                  )}
                </div>

                {/* 信息 */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium truncate">
                      {match.imageId}
                    </p>
                    {match.isMatch && (
                      <Badge variant="default" className="text-xs bg-green-600">
                        匹配
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <Progress
                      value={match.similarity * 100}
                      className="h-2 flex-1"
                    />
                    <span
                      className={cn(
                        'text-sm font-medium w-14 text-right',
                        getSimilarityColor(match.similarity)
                      )}
                    >
                      {(match.similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* 状态图标 */}
                <div className="flex-shrink-0">
                  {match.isMatch ? (
                    <CheckCircle2 className="h-5 w-5 text-green-600" />
                  ) : (
                    <XCircle className="h-5 w-5 text-muted-foreground" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
