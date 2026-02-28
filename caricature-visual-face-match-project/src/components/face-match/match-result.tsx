'use client';

import { useState, useEffect } from 'react';
import { MatchedPair } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from '@/components/ui/dialog';
import { CheckCircle2, XCircle, Trophy, ImageIcon, Maximize2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MatchResultProps {
  matches: MatchedPair[];
  queryModality: 'face' | 'caricature';
  processTime?: number;
  className?: string;
  autoOpen?: boolean; // Auto open dialog after matching
  onDialogClose?: () => void; // Callback when dialog closes
}

export function MatchResult({
  matches,
  queryModality,
  processTime,
  className,
  autoOpen = false,
  onDialogClose,
}: MatchResultProps) {
  const [showAutoDialog, setShowAutoDialog] = useState(autoOpen);

  // Update showAutoDialog when autoOpen changes
  useEffect(() => {
    if (autoOpen) {
      setShowAutoDialog(true);
    }
  }, [autoOpen]);

  // Handle dialog close
  const handleDialogClose = (open: boolean) => {
    setShowAutoDialog(open);
    if (!open && onDialogClose) {
      onDialogClose();
    }
  };

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

  // Render a single match item
  const MatchItem = ({ match, index }: { match: MatchedPair; index: number }) => (
    <div
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

      {/* 图像预览 */}
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
  );

  // Dialog content for both expand and auto-open
  const MatchDialog = ({ isOpen, onOpenChange }: { isOpen: boolean; onOpenChange: (open: boolean) => void }) => (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <span>匹配结果</span>
            <div className="flex items-center gap-2 text-sm font-normal">
              <Badge variant="outline">共 {matches.length} 条</Badge>
              {processTime !== undefined && processTime !== null && (
                <Badge variant="outline">耗时: {formatProcessTime(processTime)}</Badge>
              )}
            </div>
          </DialogTitle>
          <DialogDescription>
            查询模式: {queryModality === 'face' ? '真实人脸 → 漫画' : '漫画 → 真实人脸'}
          </DialogDescription>
        </DialogHeader>
        <div className="flex-1 overflow-y-auto pr-2" style={{ maxHeight: 'calc(80vh - 120px)' }}>
          <div className="space-y-3">
            {matches.map((match, index) => (
              <MatchItem key={match.imageId || `match-${index}`} match={match} index={index} />
            ))}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );

  return (
    <>
      {/* 主卡片 - 固定高度，内部滚动 */}
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
              {/* 展开按钮 - 始终显示 */}
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => handleDialogClose(true)}
                title="展开查看全部"
              >
                <Maximize2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            查询模式: {queryModality === 'face' ? '真实人脸 → 漫画' : '漫画 → 真实人脸'}
          </p>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden p-0 min-h-0">
          <div className="h-full overflow-y-auto px-6 pb-4" style={{ maxHeight: '300px' }}>
            <div className="space-y-3 pt-2">
              {matches.slice(0, 5).map((match, index) => (
                <MatchItem key={match.imageId || `match-${index}`} match={match} index={index} />
              ))}
              
              {/* 超过5条时显示提示 */}
              {matches.length > 5 && (
                <Button
                  variant="outline"
                  className="w-full"
                  size="sm"
                  onClick={() => handleDialogClose(true)}
                >
                  <Maximize2 className="h-4 w-4 mr-2" />
                  查看全部 {matches.length} 条结果
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 弹窗 */}
      <MatchDialog isOpen={showAutoDialog} onOpenChange={handleDialogClose} />
    </>
  );
}
